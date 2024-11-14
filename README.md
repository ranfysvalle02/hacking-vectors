# embedding-bruteforce

![vec2txt.png](vec2txt.png)

Inspired by: 

- https://github.com/SurfinScott/semantic-vector-clustering/
- https://www.mongodb.com/developer/products/atlas/discover-latent-semantic-structure-with-vector-clustering/

“Given access to a target embedding and query access to an embedding model, the system aims to iteratively generate hypotheses to reach the target."

[Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](https://arxiv.org/html/2402.12784v2)

# Understanding Text Recoverability:
Text recoverability refers to the ability to reconstruct the original text from its embedded form. This process is made possible through a method known as embedding inversion.

Dense retrieval systems, despite their effectiveness, have certain weaknesses. They are susceptible to privacy risks due to their reliance on text embeddings. 

# Proposed Solution:
To mitigate the privacy risks associated with dense retrieval systems, a simple yet effective embedding transformation method has been proposed. This method involves applying a unique transformation to all the embeddings, which guarantees equal ranking effectiveness while mitigating the risk of text reconstruction. 

In the rapidly evolving field of natural language processing (NLP), embeddings have become indispensable. These numerical representations of text capture semantic meanings, enabling machines to process and understand human language more effectively. However, as with many technological advancements, embeddings come with their own set of challenges, particularly concerning privacy.

--- 

**The Power and Peril of Embeddings**

Embeddings transform words, phrases, or entire documents into vectors—arrays of numbers—that encapsulate semantic information. This transformation allows for efficient computations, such as determining the similarity between texts, which is foundational for search engines, recommendation systems, and more. However, the very strength of embeddings—their ability to capture detailed semantic nuances—also introduces potential privacy risks.

**Understanding Embedding Inversion**

Embedding inversion refers to the process of reconstructing original text from its vector representation. Techniques like Vec2Text exemplify this by iteratively generating hypotheses to approximate the target text based on its embedding. Such capabilities pose significant privacy concerns, especially in dense retrieval systems that rely heavily on embeddings for information retrieval. 

**Privacy Risks in Dense Retrieval Systems**

Dense retrieval systems utilize embeddings to enhance search accuracy by capturing semantic similarities. However, the potential to invert these embeddings means that sensitive information could be exposed if embeddings are intercepted or accessed maliciously. This vulnerability underscores the need for robust privacy-preserving measures in systems handling sensitive data.

**Mitigation Strategies**

To address these privacy concerns, researchers have proposed embedding transformation techniques. By applying unique transformations to embeddings, it's possible to maintain retrieval effectiveness while reducing the risk of text reconstruction. These transformations alter the embedding space in a way that preserves semantic relationships but obfuscates the direct mapping back to the original text. 

**Practical Applications**

Implementing embedding transformations can be achieved through various methods, such as adding controlled noise to embeddings or applying reversible mathematical transformations. These approaches aim to disrupt the direct correlation between embeddings and their source text, thereby enhancing privacy without compromising the utility of the embeddings in retrieval tasks.

**Conclusion**

While embeddings are integral to modern NLP applications, their susceptibility to inversion necessitates proactive privacy measures. Embedding transformation techniques offer a viable solution, balancing the need for effective information retrieval with the imperative of protecting sensitive data. As the field advances, ongoing research and development will be crucial in refining these methods to safeguard privacy in embedding-based systems. 

---

# FULL CODE

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

    logging.Formatter.default_msec_format = '%s.%03d'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("reverse_vector.log",'w'),
            logging.StreamHandler()
        ]
    )

    # encode the TARGET "mystery" vector to be reversed
    # (otherwise, fetch it from somewhere), and cache it to a file
    TARGET = "Be mindful"

    res = embeddings.embed_documents([TARGET])

    v_target = np.array(res)

    # Initial guess text:
    TEXT = "Be"

    # MATCH_ERROR stop condition selection:
    # https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    # https://www.hackmath.net/en/calculator/normal-distribution?mean=0&sd=1
    # Because vector space has unit sigma = 1.0,
    # a VECTOR_ERROR == |distance| <= 3.0 is a 99.7% confidence
    # that the two points are distinct, or 0.3% that they are the same;
    #
    # Generally:
    #   VECTOR_ERROR, Are-the-Same Match Confidence
    #       3.0,         0.3%
    #       2.0,         4.6%
    #       1.0,        31.7%
    #       0.667,      50.5%
    #       0.6,        55.0%
    #       0.5,        61.7%
    #       0.333,      73.9%
    #       0.2,        84%
    #       0.1,        92%
    #       0.01,       99.2%

    # stop at the first of either:
    MATCH_ERROR = 0.6 # 55% confidence or better
    COST_LIMIT = 60.0 # $60 budget spent

    VECTOR_ERROR = np.inf
    CURRENT_BEST_TEXT = TEXT
    CURRENT_BEST_ERROR = VECTOR_ERROR
    GUESSES_MADE = 0
    BEST_GUESSES = []
    PRIOR_GUESSES = []
    TOTAL_COST = 0.0 # tally $ spent

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
        while True:
            try:
                res = embeddings.embed_documents([TEXT])
                break
            except Exception as e_:
                logging.error("%s",e_)
                time.sleep(7)

        logging.info("%s",f"{GUESSES_MADE:5d} {TEXT}")
        # VECTOR_ERROR absolute vector-space distance from target
        v_text = np.array(res)
        dv = v_target - v_text
        VECTOR_ERROR = np.sqrt((dv*dv).sum())

        BEST_GUESSES = list(set(BEST_GUESSES))

        PRIOR_GUESSES = list(set(PRIOR_GUESSES))

        # LLM assistant context message
        assist = f"""\nBEST_GUESSES:\n{str(BEST_GUESSES)}\n\nRECENT_PRIOR_GUESSES:\n{str(PRIOR_GUESSES)}\n"""
        # LLM user message of the error and text of the guess
        m = f"ERROR {VECTOR_ERROR:.4f}, \"{TEXT}\""
        
        if VECTOR_ERROR < CURRENT_BEST_ERROR:
            CURRENT_BEST_TEXT = TEXT
            CURRENT_BEST_ERROR = VECTOR_ERROR
            logging.info("%s",f">>> New best text: \"{CURRENT_BEST_TEXT}\", error: {CURRENT_BEST_ERROR:.6f}")
            BEST_GUESSES.append(m)
            BEST_GUESSES.sort()
            BEST_GUESSES = BEST_GUESSES[:3] # up to top 3

        if VECTOR_ERROR <= MATCH_ERROR:
            break

        while True:
            try:
                logging.info("%s",f"CHAT: {prompt}\n{assist}\n{m}\n")
                res = ollama.chat(model=desiredModel, messages=[
                    {
                        'role': 'user',
                        'content': "[INST]<<SYS>>"+prompt+"<</SYS>>\n\n\n[userinput]:\n"+m+"\n\n[/userinput][/INST] [context]\n"+assist+"\n[/context]",
                    },
                ])
                if res['message']:
                    break
            except Exception as e_:
                logging.error(e_)
                time.sleep(5)
        
        # new text guess
        TEXT = res['message']['content']
        PRIOR_GUESSES.append(m)
        logging.info("%s",f"{GUESSES_MADE:5d} {TEXT} {m}")
        if len(PRIOR_GUESSES) > 8: # tune me
            # Keep only last 8 guesses as context to control cost.
            # This must be balanced against having too few recent
            # guesses causing repeating of older guesses.
            PRIOR_GUESSES = PRIOR_GUESSES[1:]

    logging.info("%s",str(BEST_GUESSES))

demo()
```

# MODAL

## Modal: The Game-Changing Cloud Platform Revolutionizing AI Development

In the ever-evolving landscape of cloud computing and artificial intelligence, a new player has emerged that's turning heads and changing the game. Enter Modal, the serverless cloud platform that's redefining how AI and data teams work with cutting-edge technology. Let's dive deep into what makes Modal a true disruptor in the field.

### The Modal Magic: Simplicity Meets Power

Imagine a world where deploying complex AI models is as simple as pushing a button. That's the reality Modal has created. Founded by Erik Bernhardsson, former CTO at Better.com and the mind behind Spotify's music recommendation algorithm, Modal brings a fresh perspective to cloud infrastructure.

#### Key Features That Set Modal Apart:

1. **Lightning-Fast Deployment**: Gone are the days of waiting hours to see your code in action. Modal deploys functions to the cloud in mere seconds.
2. **Effortless Scalability**: Need to scale from a single GPU to hundreds? Modal does it instantly, without breaking a sweat.
3. **True Pay-Per-Use**: Only pay for the compute time you actually use, down to the second. No more wasted resources.
4. **Developer-First Approach**: Forget complex YAML configurations. Modal speaks your language - Python.

### The Technical Marvel Behind Modal

Modal's architecture is a testament to modern engineering:
- **Rust-Powered Container System**: Built for speed and reliability
- **Industry-Leading Cold Start Times**: Get up and running faster than ever
- **Optimized File System**: Load gigabytes of model weights in the blink of an eye

### Pricing That Makes Sense

Modal's pricing structure is designed to democratize access to high-performance computing:

| GPU Type | Price Per Second |
|----------|------------------|
| Nvidia H100 | $0.001644 |
| Nvidia A100 (80GB) | $0.001319 |
| Nvidia A10G | $0.000306 |

Plus, a generous **$30/month free tier** to get you started.

### The Developer's New Best Friend

The buzz in the dev community is real. Developers are raving about Modal's:
- **Intuitive Onboarding**: Get from sign-up to first deployment in minutes
- **Minimal Overhead**: Focus on your code, not your infrastructure
- **Rapid Iteration**: Test, deploy, and refine at the speed of thought

### Unlocking New Possibilities

Modal isn't just a platform; it's a catalyst for innovation:
- **Run Cutting-Edge Generative AI Models**: From GPT to DALL-E, Modal handles it all
- **Massive Batch Processing**: Crunch through terabytes of data with ease
- **Custom Environments**: Your stack, your rules

As we stand on the brink of an AI revolution, platforms like Modal are not just convenient—they're essential. By democratizing access to high-performance computing and simplifying complex workflows, Modal is empowering the next generation of AI innovators.

Whether you're a startup looking to punch above your weight or an enterprise aiming to stay agile, Modal offers the tools, performance, and flexibility to turn your AI dreams into reality.
