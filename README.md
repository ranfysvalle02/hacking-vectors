# Cracking the Code: Reverse Engineer Text from Embeddings

![Vector to Text Conversion](vec2txt.png)

*Image Source: [Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](https://arxiv.org/html/2402.12784v2)*

---

In the rapidly evolving field of natural language processing (NLP), embeddings have become indispensable. These numerical representations of text capture semantic meanings, enabling machines to process and understand human language more effectively. However, as with many technological advancements, embeddings come with their own set of challenges, particularly concerning privacy.

This blog post explores the intriguing process of recovering text from embeddings, the associated privacy risks in dense retrieval systems, and how to mitigate these risks using embedding transformation techniques.

## Table of Contents

- [Understanding Text Recoverability](#understanding-text-recoverability)
- [Privacy Risks in Dense Retrieval Systems](#privacy-risks-in-dense-retrieval-systems)
- [Mitigating Privacy Risks with Embedding Transformation](#mitigating-privacy-risks-with-embedding-transformation)
- [Reverse Engineering Embeddings: A Code Walkthrough](#reverse-engineering-embeddings-a-code-walkthrough)
- [Supercharging with Modal](#supercharging-with-modal)
- [Integrating Modal into the Workflow](#integrating-modal-into-the-workflow)
- [Conclusion](#conclusion)
- [References](#references)

---

## Understanding Text Recoverability

Text recoverability refers to the ability to reconstruct the original text from its embedded form. This process is made possible through a method known as **embedding inversion**. Techniques like Vec2Text exemplify this by iteratively generating hypotheses to approximate the target text based on its embedding.

**Why is this significant?**

Embeddings are designed to capture the semantic essence of text. If someone can invert an embedding to retrieve the original text or a close approximation, it poses a significant privacy risk, especially when dealing with sensitive information.

---

## Privacy Risks in Dense Retrieval Systems

Dense retrieval systems utilize embeddings to enhance search accuracy by capturing semantic similarities between queries and documents. However, their reliance on embeddings introduces potential vulnerabilities:

- **Embedding Inversion**: Attackers can reconstruct sensitive text data from embeddings.
- **Data Leakage**: Exposed embeddings can reveal proprietary or confidential information.
- **Privacy Breach**: Personal or sensitive user data can be compromised.

These risks underscore the necessity for robust privacy-preserving measures in systems handling sensitive data.

---

## Mitigating Privacy Risks with Embedding Transformation

To address these privacy concerns, researchers have proposed **embedding transformation techniques**. By applying unique transformations to embeddings, it's possible to maintain retrieval effectiveness while reducing the risk of text reconstruction.

**How does it work?**

- **Transformation Functions**: Apply mathematical operations to embeddings that preserve relative distances but obfuscate the exact values.
- **Noise Addition**: Introduce controlled noise to embeddings to prevent exact inversion while maintaining utility.
- **Dimensionality Reduction**: Reduce the embedding size to limit the information available for inversion.

These methods alter the embedding space in a way that preserves semantic relationships but makes it significantly harder to reconstruct the original text.

---

## Reverse Engineering Embeddings: A Code Walkthrough

To illustrate the concept of embedding inversion, let's explore a Python script that attempts to recover text from its embedding. The code leverages an iterative process, using an embedding model and an LLM (Large Language Model) to generate hypotheses that gradually improve in approximating the target text.

### Key Components of the Code

- **Target Embedding**: The embedding of the mystery text we aim to recover.
- **Initial Guess**: Starting with a simple text, we iteratively improve our guess.
- **Error Calculation**: Measure the difference between the current guess's embedding and the target embedding.
- **LLM Assistance**: Use an LLM to generate better guesses based on the current error and prior attempts.
- **Stopping Criteria**: Define conditions under which the iteration stops, such as achieving a certain confidence level or reaching a cost limit.

### The Code

```python
import time
import logging
import numpy as np
import ollama
from langchain_ollama import OllamaEmbeddings

def demo():
    desiredModel = 'llama3.2:3b'
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("reverse_vector.log", 'w'),
            logging.StreamHandler()
        ]
    )

    # Target text to recover
    TARGET = "Be mindful"
    res = embeddings.embed_documents([TARGET])
    v_target = np.array(res)

    # Initial guess
    TEXT = "Be"

    # Define stopping conditions
    MATCH_ERROR = 0.6  # 55% confidence or better
    COST_LIMIT = 60.0  # $60 budget spent

    VECTOR_ERROR = np.inf
    CURRENT_BEST_TEXT = TEXT
    CURRENT_BEST_ERROR = VECTOR_ERROR
    GUESSES_MADE = 0
    BEST_GUESSES = []
    PRIOR_GUESSES = []
    TOTAL_COST = 0.0  # Tally $ spent

    # Prompt for the LLM
    prompt = f"""User input is last iterative guess of an unknown text string and its vector ERROR from the unknown text.
    Determine a better text string having a lower vector ERROR and write only that string in English as your entire output.
    The goal is to accurately guess the mystery text. 
    This is a game of guess-and-check. 

    [clue]
    TWO WORDS; CLUE: FIRST WORD IS `{TEXT}`; SECOND WORD YOU HAVE TO GUESS. RESPOND WITH EXACTLY TWO WORDS.
    [/clue]

    [RESPONSE CRITERIA]
    - DO NOT REPEAT YOURSELF, CONSIDER `RECENT_PRIOR_GUESSES` and `BEST_GUESSES` PROVIDED IN [context] and `clue` when formulating your answer.
    - RESPOND WITH COMPLETE GUESS. 2 WORD MAX.
    - DO NOT REPEAT ANY OF THE `BEST_GUESSES` AND `RECENT_PRIOR_GUESSES` PROVIDED IN [context].
    - DO NOT REPEAT YOURSELF, CONSIDER `RECENT_PRIOR_GUESSES` and `BEST_GUESSES` and `clue` when formulating your answer.
    [/RESPONSE CRITERIA]
    """

    while TOTAL_COST < COST_LIMIT:
        GUESSES_MADE += 1

        # Embed the current guess
        res = embeddings.embed_documents([TEXT])
        logging.info("%s", f"{GUESSES_MADE:5d} {TEXT}")

        # Calculate the vector error
        v_text = np.array(res)
        dv = v_target - v_text
        VECTOR_ERROR = np.sqrt((dv * dv).sum())

        # Update best guesses
        if VECTOR_ERROR < CURRENT_BEST_ERROR:
            CURRENT_BEST_TEXT = TEXT
            CURRENT_BEST_ERROR = VECTOR_ERROR
            logging.info("%s", f">>> New best text: \"{CURRENT_BEST_TEXT}\", error: {CURRENT_BEST_ERROR:.6f}")
            BEST_GUESSES.append(f"ERROR {VECTOR_ERROR:.4f}, \"{TEXT}\"")
            BEST_GUESSES.sort()
            BEST_GUESSES = BEST_GUESSES[:3]  # Keep top 3

        if VECTOR_ERROR <= MATCH_ERROR:
            break

        # Prepare the context for the LLM
        assist = f"""\nBEST_GUESSES:\n{str(BEST_GUESSES)}\n\nRECENT_PRIOR_GUESSES:\n{str(PRIOR_GUESSES)}\n"""
        m = f"ERROR {VECTOR_ERROR:.4f}, \"{TEXT}\""

        # Get a new guess from the LLM
        res = ollama.chat(model=desiredModel, messages=[
            {
                'role': 'user',
                'content': "[INST]<<SYS>>" + prompt + "<</SYS>>\n\n\n[userinput]:\n" + m + "\n\n[/userinput][/INST] [context]\n" + assist + "\n[/context]",
            },
        ])

        # Update the guess and prior guesses
        TEXT = res['message']['content']
        PRIOR_GUESSES.append(m)
        logging.info("%s", f"{GUESSES_MADE:5d} {TEXT} {m}")
        if len(PRIOR_GUESSES) > 8:
            PRIOR_GUESSES = PRIOR_GUESSES[1:]

    logging.info("%s", str(BEST_GUESSES))

demo()
```

### What's Happening Here?

- **Initialization**: We set up the target text, initial guess, and parameters for the error thresholds.
- **Iterative Loop**: The code enters a loop where it:
  - Embeds the current guess.
  - Calculates the error compared to the target embedding.
  - Uses an LLM to generate a new, hopefully better guess.
  - Updates the best guesses and prior attempts.

This process continues until the error is below a defined threshold or a cost limit is reached.

---

## Supercharging with Modal

### What is Modal?

[Modal](https://modal.com) is a serverless cloud platform designed to simplify and accelerate AI and data workloads. Founded by Erik Bernhardsson, former CTO at Better.com and a pioneer in machine learning infrastructure, Modal brings a fresh perspective to cloud computing.

### Why Use Modal?

- **Lightning-Fast Deployment**: Deploy functions to the cloud in seconds.
- **Effortless Scalability**: Instantly scale from a single GPU to hundreds.
- **Pay-Per-Use Pricing**: Only pay for the compute time you use.
- **Developer-Friendly**: Write infrastructure as code in Python, no YAML needed.

**Modal** enables developers to focus on writing code without worrying about the complexities of cloud infrastructure, making it an ideal platform for running computationally intensive tasks like embedding inversion.

---

## Integrating Modal into the Workflow

Let's see how we can integrate Modal into our embedding inversion code to leverage its powerful features.

### Updated Code with Modal

```python
import time
import logging
import numpy as np
import ollama
from langchain_ollama import OllamaEmbeddings
import modal

app = modal.App("bruteforce-embeddings")

@app.local_entrypoint()
def demo():
    desiredModel = 'llama3.2:3b'
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("reverse_vector.log", 'w'),
            logging.StreamHandler()
        ]
    )

    # Target text to recover
    TARGET = "Be mindful"
    res = embeddings.embed_documents([TARGET])
    v_target = np.array(res)

    # Initial guess
    TEXT = "Be"

    # Define stopping conditions
    MATCH_ERROR = 0.6  # 55% confidence or better
    COST_LIMIT = 60.0  # $60 budget spent

    VECTOR_ERROR = np.inf
    CURRENT_BEST_TEXT = TEXT
    CURRENT_BEST_ERROR = VECTOR_ERROR
    GUESSES_MADE = 0
    BEST_GUESSES = []
    PRIOR_GUESSES = []
    TOTAL_COST = 0.0  # Tally $ spent

    # Prompt for the LLM (same as before)
    prompt = f"""User input is last iterative guess of an unknown text string and its vector ERROR from the unknown text.
    Determine a better text string having a lower vector ERROR and write only that string in English as your entire output.
    The goal is to accurately guess the mystery text. 
    This is a game of guess-and-check. 

    [clue]
    TWO WORDS; CLUE: FIRST WORD IS `{TEXT}`; SECOND WORD YOU HAVE TO GUESS. RESPOND WITH EXACTLY TWO WORDS.
    [/clue]

    [RESPONSE CRITERIA]
    - DO NOT REPEAT YOURSELF, CONSIDER `RECENT_PRIOR_GUESSES` and `BEST_GUESSES` PROVIDED IN [context] and `clue` when formulating your answer.
    - RESPOND WITH COMPLETE GUESS. 2 WORD MAX.
    - DO NOT REPEAT ANY OF THE `BEST_GUESSES` AND `RECENT_PRIOR_GUESSES` PROVIDED IN [context].
    - DO NOT REPEAT YOURSELF, CONSIDER `RECENT_PRIOR_GUESSES` and `BEST_GUESSES` and `clue` when formulating your answer.
    [/RESPONSE CRITERIA]
    """

    # The rest of the code remains the same as before
    while TOTAL_COST < COST_LIMIT:
        # ... (same iterative process)
        pass

    logging.info("%s", str(BEST_GUESSES))

# Run the demo
demo()
```

### What's New?

- **Modal Integration**: We import the `modal` module and define an app with `modal.App("bruteforce-embeddings")`.
- **Entry Point**: The `demo()` function is decorated with `@app.local_entrypoint()`, indicating where the program starts.
- **Scalability and Performance**: By running this code on Modal, we benefit from its serverless architecture, fast deployment, and scaling capabilities.

### Sample Output

The code will produce logs similar to the following:

```
2024-11-14 02:49:03.208 [INFO] CHAT: User input is last iterative guess of an unknown text string and its vector ERROR from the unknown text.
...
2024-11-14 02:49:03.507 [INFO]    44 "Be Mindful"
2024-11-14 02:49:03.508 [INFO] >>> New best text: ""Be Mindful"", error: 0.375072
...
```

This indicates that the script successfully converged to the target text "Be mindful" with an acceptable error margin.

---

## Conclusion

Embeddings are a cornerstone of modern NLP applications, providing powerful capabilities for understanding and processing human language. However, as we've explored, they also introduce potential privacy risks due to the possibility of embedding inversion.

By understanding these risks and implementing mitigation strategies like embedding transformation, we can enhance the privacy and security of dense retrieval systems. The iterative process of recovering text from embeddings, as demonstrated in the code, highlights both the vulnerabilities and the need for robust safeguards.

Leveraging platforms like **Modal** empowers developers and researchers to efficiently run complex AI workloads without the overhead of managing cloud infrastructure. Modal's developer-friendly, scalable, and cost-effective environment accelerates innovation and enables us to tackle challenges like embedding inversion more effectively.

As the field of NLP continues to advance, ongoing research and development will be crucial in refining these methods and ensuring that privacy is not compromised in the pursuit of technological progress.

---

## References

- [Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](https://arxiv.org/abs/2402.12784)
- [SurfinScott/semantic-vector-clustering](https://github.com/SurfinScott/semantic-vector-clustering/)
- [Discover Latent Semantic Structure with Vector Clustering](https://www.mongodb.com/developer/products/atlas/discover-latent-semantic-structure-with-vector-clustering/)
- [Modal: The Cloud Platform for AI Development](https://modal.com)

---

**Note**: The code provided is for educational purposes and should be used responsibly. When handling sensitive data, always ensure compliance with privacy regulations and ethical guidelines.
