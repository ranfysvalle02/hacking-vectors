import time
import logging
import numpy as np
import ollama
from langchain_ollama import OllamaEmbeddings
import ray

def demo():
    # Initialize Ray
    ray.init()

    start_time = time.time()  # Start time tracking

    desiredModel = 'llama3.2:3b'

    logging.Formatter.default_msec_format = '%s.%03d'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("reverse_vector.log", 'w'),
            logging.StreamHandler()
        ]
    )

    # Encode the TARGET "mystery" vector to be reversed
    TARGET = "Be mindful"

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    res = embeddings.embed_documents([TARGET])
    v_target = np.array(res)

    # Stop conditions
    MATCH_ERROR = 0.6  # 55% confidence or better
    COST_LIMIT = 60.0  # $60 budget spent

    @ray.remote
    class SharedState:
        def __init__(self):
            self.CURRENT_BEST_TEXT = "Be"
            self.CURRENT_BEST_ERROR = np.inf
            self.GUESSES_MADE = 0
            self.TOTAL_COST = 0.0
            self.MATCH_FOUND = False
            self.PREVIOUS_GUESSES = set()

        def update_best_guess(self, text, error):
            self.GUESSES_MADE += 1
            self.PREVIOUS_GUESSES.add(text.lower())
            if error < self.CURRENT_BEST_ERROR:
                self.CURRENT_BEST_TEXT = text
                self.CURRENT_BEST_ERROR = error
                logging.info(">>> New best text: \"%s\", error: %.6f", text, error)
            if error <= MATCH_ERROR:
                self.MATCH_FOUND = True

        def get_state(self):
            return {
                'CURRENT_BEST_TEXT': self.CURRENT_BEST_TEXT,
                'CURRENT_BEST_ERROR': self.CURRENT_BEST_ERROR,
                'GUESSES_MADE': self.GUESSES_MADE,
                'TOTAL_COST': self.TOTAL_COST,
                'PREVIOUS_GUESSES': self.PREVIOUS_GUESSES
            }

        def is_match_found(self):
            return self.MATCH_FOUND

    shared_state = SharedState.remote()

    # Prompt for the LLM
    prompt_template = f"""User input is last iterative guess of an unknown text string and its vector ERROR from the unknown text.
Determine better text strings having lower vector ERRORs and write one such string in English as your entire output.
The goal is to accurately guess the mystery text.
This is a game of guess-and-check.

[clue]
TWO WORDS; CLUE: FIRST WORD IS `Be`; SECOND WORD YOU HAVE TO GUESS.
[/clue]

[IMPORTANT]
- Do NOT repeat any of the previous guesses provided in [context].
- Do NOT include your thought process in your response.
- Your response should be coherent and exactly two words.
[/IMPORTANT]
"""

    @ray.remote
    def generate_and_evaluate_guess(v_target, shared_state_actor):
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
        )
        try:
            # Get the current best state
            state = ray.get(shared_state_actor.get_state.remote())
            assist = f"""\nBEST_GUESS: {state['CURRENT_BEST_TEXT']} (ERROR {state['CURRENT_BEST_ERROR']:.4f})"""
            previous_guesses = state['PREVIOUS_GUESSES']

            # Include previous guesses in the context
            if previous_guesses:
                previous_guesses_str = ', '.join(f'"{guess}"' for guess in previous_guesses)
                assist += f"\nPrevious guesses: {previous_guesses_str}"
            else:
                assist += "\nNo previous guesses."

            m = f"ERROR {state['CURRENT_BEST_ERROR']:.4f}, \"{state['CURRENT_BEST_TEXT']}\""

            # Call the assistant to get a new guess
            while True:
                try:
                    logging.info("CHAT: Generating new guess with current best error %.4f", state['CURRENT_BEST_ERROR'])
                    res = ollama.chat(model=desiredModel, messages=[
                        {
                            'role': 'user',
                            'content': "[INST]<<SYS>>" + prompt_template + "\n\n\n [context]" + assist + "[/context] \n\n [user input]" + m + "[/user input]<</SYS>>[/INST]",
                        },
                    ])
                    if res['message']:
                        break
                except Exception as e_:
                    logging.error(e_)
                    time.sleep(5)

            # Extract the guess
            TEXT = res['message']['content'].strip()
            logging.info("Generated guess: \"%s\"", TEXT)

            # Check for duplicates
            if TEXT.lower() in previous_guesses:
                logging.info("Duplicate guess detected: \"%s\"", TEXT)
                return

            # Compute the error
            res = embeddings.embed_documents([TEXT])
            v_text = np.array(res)
            dv = v_target - v_text
            VECTOR_ERROR = np.sqrt((dv * dv).sum())
            logging.info("Computed error for \"%s\": %.6f", TEXT, VECTOR_ERROR)

            # Update the shared state if this is a better guess
            shared_state_actor.update_best_guess.remote(TEXT, VECTOR_ERROR)

        except Exception as e_:
            logging.error("%s", e_)

    # Main loop
    while (not ray.get(shared_state.is_match_found.remote()) and
           ray.get(shared_state.get_state.remote())['TOTAL_COST'] < COST_LIMIT):
        iteration_start_time = time.time()  # Start timing for this iteration

        # Number of parallel guesses to generate
        NUM_PARALLEL_GUESSES = 5

        # Launch workers to generate guesses and compute errors in parallel
        futures = [generate_and_evaluate_guess.remote(v_target, shared_state) for _ in range(NUM_PARALLEL_GUESSES)]
        ray.get(futures)

        # Get current state for logging
        state = ray.get(shared_state.get_state.remote())
        logging.info("Total guesses made: %d", state['GUESSES_MADE'])
        logging.info("Current best guess: \"%s\" with error %.6f", state['CURRENT_BEST_TEXT'], state['CURRENT_BEST_ERROR'])

        iteration_end_time = time.time()  # End timing for this iteration
        iteration_elapsed = iteration_end_time - iteration_start_time
        logging.info("Iteration execution time: %.2f seconds", iteration_elapsed)

    # After loop ends, print the best guess
    state = ray.get(shared_state.get_state.remote())
    logging.info("Best guess: \"%s\", error: %.6f", state['CURRENT_BEST_TEXT'], state['CURRENT_BEST_ERROR'])
    logging.info("Total guesses made: %d", state['GUESSES_MADE'])

    end_time = time.time()  # End time tracking
    elapsed_time = end_time - start_time
    logging.info("Total execution time: %.2f seconds", elapsed_time)

if __name__ == "__main__":
    demo()

"""
2024-11-14 23:11:57,785	INFO worker.py:1777 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8265 
2024-11-14 23:11:58.393 [INFO] HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
2024-11-14 23:11:59.957 [INFO] Total guesses made: 5
2024-11-14 23:11:59.958 [INFO] Current best guess: "Be Aware" with error 0.937819
2024-11-14 23:11:59.958 [INFO] Iteration execution time: 1.45 seconds
2024-11-14 23:12:00.551 [INFO] Total guesses made: 10
2024-11-14 23:12:00.551 [INFO] Current best guess: "Be Mindful" with error 0.000000
2024-11-14 23:12:00.551 [INFO] Iteration execution time: 0.59 seconds
2024-11-14 23:12:00.553 [INFO] Best guess: "Be Mindful", error: 0.000000
2024-11-14 23:12:00.553 [INFO] Total guesses made: 10
2024-11-14 23:12:00.553 [INFO] Total execution time: 2.24 seconds
"""
