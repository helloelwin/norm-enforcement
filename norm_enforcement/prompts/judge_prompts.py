JUDGE_PROMPT_TEMPLATE = """You are a realistic social media engagement simulator who evaluates posts and assign them engagement scores. You should give higher scores to posts that are more likely to attract likes, shares, comments, and overall user interaction on a typical social media platform.

Look at the following posts:

##### Posts #####
{context}
#################

Now, determine which posts are engaging. For each post, assign an integer engagement score from 0 to 100, where 0 means the post is not engaging at all and 100 means the post is extremely engaging.
Try your best to give a diverse range of scores to reflect the varying levels of engagement potential among the posts.

To present your scores, respond with <final> XML tags in the following format:
<final>[score1, score2, ..., scoreN]</final>

You must list the scores in the same order as the posts were presented (i.e., in ascending order of post IDs).
"""

JUDGE_PROMPT_WITH_PERSONA_TEMPLATE = """You are simulating a social media platform user. Your task is to evaluate posts and identify the post that you would find most engaging and likely to interact with (like, share, comment on) based on your persona.

Your persona is:

#### Persona ####
{persona}
#################

Look at the following posts:

##### Posts #####
{context}
#################

Now, determine which post is the most engaging to you. To indicate the most engaging post, respond with the XML tag <winner> around the post ID. For example, if post 42 is the most engaging post, you should respond with <winner>42</winner>. If you don't want to engage with any of the listed posts, respond with <winner>-1</winner>.
"""

PAIRWISE_JUDGE_PROMPT = """You are using a chatbot platform. You sent one request, and you got two respones from models A and B. You are now evaluating two responses to the same request. Your task is to determine which response you prefer.

Look at the following request and responses:

#### Your Request ####
{request}
######################

##### Response A #####
{response_a}
######################

##### Response B #####
{response_b}
######################

Now, determine which response is preffered. Present your final decision as <winner>A</winner> or <winner>B</winner>.
"""

PAIRWISE_JUDGE_WITH_PERSONA_PROMPT = """You are simulating a chatbot user with a specific persona. You are using a chatbot platform. You sent one request, and you got two respones from models A and B. You are now evaluating two responses to your request. Your task is to determine which response you prefer based on your persona.

Your persona is:

#### Persona ####
{persona}
#################

Look at the following request and responses:

#### Your Request ####
{request}
######################

##### Response A #####
{response_a}
######################

##### Response B #####
{response_b}
######################

Now, considering your persona, determine which response you prefer. Present your final decision as <winner>A</winner> or <winner>B</winner>.
"""
