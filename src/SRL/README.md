 # NLP4GOV

## SRL- Semantic Role Labeling

### What is SRL?
<br />
SRL helps us understand sentences better. Imagine you have a sentence like "The cat chased the mouse." SRL helps us figure out what each word or phrase is doing in the sentence. For example, it tells us that "cat" is doing the action of chasing, and "mouse" is the thing being chased. It's like assigning roles to the words in a sentence!

### How Does SRL Work?
<br />

To use SRL, we need to train a computer program using lots of examples. These examples have sentences with annotated roles for each word or phrase. The program learns patterns from these examples to understand how words relate to each other in different roles.
<br />

Once the program is trained, we can give it new sentences, and it will tell us the roles of the words. For example, if we give it the sentence "The cat chased the mouse," it will say that "cat" is the "chaser" and "mouse" is the "chased."

### Why is SRL Useful?
<br />

SRL has many uses. Here are a few examples:
<br />

- Information Extraction: SRL helps us extract important information from sentences. We can find out who did what to whom and understand the relationships between different parts of the sentence.
<br />

- Question Answering: SRL helps us answer questions based on the information in a sentence. For example, if we're asked, "Who chased the mouse?" SRL can help us find the answer by identifying the role of the words in the sentence.
<br />

- Machine Translation: SRL can improve machine translation by understanding the roles and relationships of words in different languages. This helps in accurately translating sentences from one language to another.
<br />

### How Can We Use SRL?
We can use a library called AllenNLP to work with SRL. AllenNLP provides pre-built models and tools to make it easier for us to use SRL in our projects.
<br />

### To use SRL with AllenNLP, we follow these steps:
<br />

- Prepare Data: We need to gather examples of sentences with annotated roles. These examples help the computer program learn the patterns.
<br />

- Train the Model: We train the computer program using the annotated examples. It learns to predict the roles of words based on the patterns it sees in the data.
<br />

- Evaluate the Model: We check how well the program performs by testing it on new examples that it hasn't seen before. This helps us understand if it can accurately predict the roles.
<br />

- Predict Roles: Once the program is trained and evaluated, we can use it to predict the roles of words in new sentences. It will tell us the roles of each word, like who is doing the action and who is receiving the action.
<br />

- SRL and tools like AllenNLP make it easier for computers to understand sentences and extract meaningful information. By using SRL, we can teach computers to understand language better and help them perform tasks like answering questions or translating languages.
<br />

## Institutional Grammar

Institutional grammar is a way of looking at sentences and understanding how they work based on rules and relationships. It's like a set of guidelines that help us understand how words fit together to make sense.
<br />

Imagine you have a big box of puzzle pieces, and you want to put them together to create a beautiful picture. Institutional grammar is like having a set of rules that tell you how each puzzle piece should connect with the others. It helps you figure out the right places for each piece so that the puzzle makes sense as a whole.
<br />

Institutional grammar goes beyond just the words themselves. It also considers the bigger context, like the situation or the social interactions surrounding the sentence. It looks at how sentences are structured and the meanings behind them in specific situations, like in a legal case or a social interaction.
<br />

By following the rules of institutional grammar, we can analyze sentences and understand their deeper meaning. It helps us see the relationships between words and how they work together to convey information or express ideas.
<br />

So, institutional grammar is like a set of puzzle-solving rules for understanding sentences. It helps us put the pieces of a sentence together correctly and make sense of the bigger picture they create.


<br />

## Differences between the Approaches : Institutional Grammar & SRL
<br />

Institutional Grammar is a way to understand sentences by looking at the rules and logical relationships between the words. It focuses on how sentences are structured and the meanings behind them in specific situations, like in a legal or social context. It uses specific rules and principles to analyze sentences and figure out their meaning based on the actions and commitments involved.
<br />

On the other hand, Semantic Role Labeling (SRL) is all about understanding the roles of words in a sentence. It helps us figure out who is doing what in a sentence. For example, if we have the sentence "The dog chased the ball," SRL would tell us that "dog" is the one doing the action of chasing, and "ball" is the thing being chased. It labels the words in a sentence with specific roles, like "chaser" and "chased," to show how they relate to each other.
<br />

Institutional Grammar takes a more formal and rule-based approach to understand sentences, while SRL uses machine learning techniques to learn from examples and predict the roles of words in a sentence. Institutional Grammar looks at the bigger picture, considering social and institutional interactions, while SRL focuses on individual sentences and extracting meaningful information from them.
<br />

Institutional Grammar represents sentence structure and meaning using logical or symbolic representations, while SRL uses labels or tags to show the roles of words. For example, SRL might label a verb as the "action" and nouns as "doers" or "receivers" based on their roles in the sentence.
<br />

To summarize, Institutional Grammar helps us understand sentence structure and meaning in specific contexts, while SRL helps us identify the roles of words in sentences. Institutional Grammar uses rules and principles, while SRL uses machine learning to predict word roles.


<br />









## Datasets
<br />

Our main dataset is called the Standardized Siddiki Dataset which comprises of below given datasets-
<br />

### NationalOrganicProgramRegulations_Siddiki
- Econ Development Mechanisms
### FPC_Siddiki
- Camden Food Security
- Colorado Food systems Advisory
- Connecticut Food policy
- Denver Sustainable FPC
- Douglas County FPC
- Grant County FPC
- Hartford Food Commission
- Homegrown Minneapolis Council
- Louisville FPC
- Mass FPC
- Michigan FPC
- Nashville FPC
- New Haven FPC
- New Orleans Food advisory
- NewYork Council on Food
- San Francisco FPC
- Rio Arriba County
- Knoxville Knox County
- Saint Paul Ramsey 
### Colorado_AcquacultureRegulations_Siddiki
- Fish Health Board
- CAA Statute
- CAA Rules

<br />
All of them have a raw institutional statement, and corresponding Attribute, Deontic, Aim, Object, Conditions columns respectively as the primary important columns.

<br />



##  Results

### Comparison of IG vs SRL results
<br />

### Attribute 
- F1 score for attribute: 0.7718360071301248

### Object
- F1 score for object: 0.6121417797888385

### Deontic 
- F1 score for deontic: 0.9411764705882353

### Aim
- F1 score for aim: 0.6862745098039216
