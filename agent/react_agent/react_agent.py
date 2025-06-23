import re

from openai import OpenAI

DEFAULT_MODEL = ""
api_key = ""
base_url = ""

client = OpenAI(api_key=api_key, base_url=base_url)


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.message = []
        if self.system:
            self.message.append({"role": "system", "content": system})

    def invoke(self, message):
        self.message.append({"role": "user", "content": message})
        result = self.execute()
        self.message.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=self.message,
            temperature=0.9,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=["\n\n"],
        )
        return completion.choices[0].message.content


prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

ask_fruit_unit_price:
e.g. ask_fruit_unit_price: apple
Asks the user for the price of a fruit

Example session:

Question: What is the unit price of apple?
Thought: I need to ask the user for the price of an apple to provide the unit price. 
Action: ask_fruit_unit_price: apple
PAUSE

You will be called again with this:

Observation: Apple unit price is 10/kg

You then output:

Answer: The unit price of apple is 10 per kg.
""".strip()


def calculate(what: str):
    return eval(what)


def ask_fruit_unit_price(fruit: str):
    if fruit.casefold() == "apple":
        return "Apple unit price is 10/kg"
    elif fruit.casefold() == "banana":
        return "Orange unit price is 8/kg"
    else:
        return "{} unit price is 20/kg".format(fruit)
action_re = re.compile(r'^Action: (\w+): (.*)$')
know_actions = {
    "calculate": calculate,
    "ask_fruit_unit_price": ask_fruit_unit_price
}


def query(question,max_turns=5):
    i = 0
    agent = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        response = agent.invoke(next_prompt)
        print( response)
        actions = [action_re.match(a) for a in response.split('\n') if action_re.match(a)]
        if actions:
            action,action_input = actions[0].groups()
            if action not in know_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print("---running {} {}".format(action, action_input))
            observation = know_actions[action](action_input)
            print("observation: {}".format(observation))
            next_prompt = "Observation: {}".format( observation)
        else:
            return

if __name__ == '__main__':
    query("calculate: 1 * 10 + 2 * 6")