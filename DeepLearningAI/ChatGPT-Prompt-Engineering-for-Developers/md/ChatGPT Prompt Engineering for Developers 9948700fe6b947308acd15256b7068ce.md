# ChatGPT Prompt Engineering for Developers

[DLAI - Learning Platform Beta](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/2/guidelines)

## 1. Guidelines

- Principle 1: Write Clear and specific instructions
    - ****Tactic 1: Use delimiters to clearly indicate distinct parts of the input****
        - Delimiters can be anything like: ```, """, < >, `<tag> </tag>`
            
            ```python
            prompt = f"""
            Summarize the text delimited by triple backticks \ 
            into a single sentence.
            ```{text}```
            """
            ```
            
    - ****Tactic 2: Ask for a structured output****
        - JSON, HTML, XML
            
            ```python
            prompt = f"""
            Generate a list of three made-up book titles along \ 
            with their authors and genres. 
            Provide them in JSON format with the following keys: 
            book_id, title, author, genre.
            """
            ```
            
    - ****Tactic 3: Ask the model to check whether conditions are satisfied****
        
        ```python
        prompt = f"""
        You will be provided with text delimited by triple quotes. 
        If it contains a sequence of instructions, \ 
        re-write those instructions in the following format:
        
        Step 1 - ...
        Step 2 - …
        …
        Step N - …
        
        If the text does not contain a sequence of instructions, \ 
        then simply write \"No steps provided.\"
        
        \"\"\"{text_1}\"\"\"
        """
        ```
        
    - ****Tactic 4: "Few-shot" prompting**** 🗯️ 给出正确的示范，但你想要gpt按照某种语气某种文风说话的时候，很有用，比用一堆形容词管用
        
        ```python
        prompt = f"""
        Your task is to answer in a consistent style.
        
        <child>: Teach me about patience.
        
        <grandparent>: The river that carves the deepest \ 
        valley flows from a modest spring; the \ 
        grandest symphony originates from a single note; \ 
        the most intricate tapestry begins with a solitary thread.
        
        <child>: Teach me about resilience.
        """
        ```
        

![截屏2023-06-25 03.09.32.png](ChatGPT%20Prompt%20Engineering%20for%20Developers%209948700fe6b947308acd15256b7068ce/%25E6%2588%25AA%25E5%25B1%258F2023-06-25_03.09.32.png)

---

- ****Principle 2: Give the model time to “think”****
    - ****Tactic 1: Specify the steps required to complete a task, Ask for output in a specified format****
        
        ```python
        prompt_2 = f"""
        Your task is to perform the following actions: 
        1 - Summarize the following text delimited by 
          <> with 1 sentence.
        2 - Translate the summary into French.
        3 - List each name in the French summary.
        4 - Output a json object that contains the 
          following keys: french_summary, num_names.
        
        Use the following format:
        Text: <text to summarize>
        Summary: <summary>
        Translation: <summary translation>
        Names: <list of names in Italian summary>
        Output JSON: <json with summary and num_names>
        
        Text: <{text}>
        """
        ```
        
    - ****Tactic 2: Instruct the model to work out its own solution before rushing to a conclusion****
        
        ```python
        prompt = f"""
        Your task is to determine if the student's solution \
        is correct or not.
        To solve the problem do the following:
        - First, work out your own solution to the problem. 
        - Then compare your solution to the student's solution \ 
        and evaluate if the student's solution is correct or not. 
        Don't decide if the student's solution is correct until 
        you have done the problem yourself.
        
        Use the following format:
        Question:
        ```
        question here
        ```
        Student's solution:
        ```
        student's solution here
        ```
        Actual solution:
        ```
        steps to work out the solution and your solution here
        ```
        Is the student's solution the same as actual solution \
        just calculated:
        ```
        yes or no
        ```
        Student grade:
        ```
        correct or incorrect
        ```
        
        Question:
        ```
        I'm building a solar power installation and I need help \
        working out the financials. 
        - Land costs $100 / square foot
        - I can buy solar panels for $250 / square foot
        - I negotiated a contract for maintenance that will cost \
        me a flat $100k per year, and an additional $10 / square \
        foot
        What is the total cost for the first year of operations \
        as a function of the number of square feet.
        ``` 
        Student's solution:
        ```
        Let x be the size of the installation in square feet.
        Costs:
        1. Land cost: 100x
        2. Solar panel cost: 250x
        3. Maintenance cost: 100,000 + 100x
        Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
        ```
        Actual solution:
        """
        ```
        
        ---
        
        ****Model Limitations: Hallucinations****
        
        避免编造，要求从给定文档中寻找，给出应用，明确不要编造，just idont know
        
        ---
        
        ## 2. Iterative
        
        Iterative Prompt Development
        
        使用characters来控制长度，tokenizer
        
        通过****Generate a marketing product description from a product fact sheet，一步步优化：****
        
        限制长度（too long） ⇒ 要求是focus on参数（****Text focuses on the wrong details，****Ask it to focus on the aspects that are relevant to the intended audience.）
        
        优质的prompt也是这样iterative出来的
        
        因此，没必要需要知道太多现成的prompt，适合自己应用的才是最好的，需要反复迭代
        
        项目大的时候，可以通过通过大量数据来评估不同prompt的效果
        
        ---
        
        ![截屏2023-06-25 03.32.59.png](ChatGPT%20Prompt%20Engineering%20for%20Developers%209948700fe6b947308acd15256b7068ce/%25E6%2588%25AA%25E5%25B1%258F2023-06-25_03.32.59.png)
        
        ---
        
        ## Summarizing
        
        没啥
        
        ****Text to summarize****
        
        ```python
        prompt = f"""
        Your task is to generate a short summary of a product \
        review from an ecommerce site. 
        
        Summarize the review below, delimited by triple 
        backticks, in at most 30 words. 
        
        Review: ```{prod_review}```
        """
        ```
        
        ****Summarize with a focus on shipping and delivery****
        
        ```python
        prompt = f"""
        Your task is to generate a short summary of a product \
        review from an ecommerce site to give feedback to the \
        pricing deparmtment, responsible for determining the \
        price of the product.  
        
        Summarize the review below, delimited by triple 
        backticks, in at most 30 words, and focusing on any aspects \
        that are relevant to the price and perceived value. 
        
        Review: ```{prod_review}```
        """
        ```
        
        ### Summaries include topics that are not related to the topic of focus.
        
        ****Try "extract" instead of "summarize"****
        
        ```python
        prompt = f"""
        Your task is to extract relevant information from \ 
        a product review from an ecommerce site to give \
        feedback to the Shipping department. 
        
        From the review below, delimited by triple quotes \
        extract the information relevant to shipping and \ 
        delivery. Limit to 30 words. 
        
        Review: ```{prod_review}```
        """
        ```
        
        ---
        
        ## Inferring
        
        首要的就是情感分析，然后提取特定信息：从评论中提取产品名称和品牌
        
        ```python
        prompt = f"""
        Identify the following items from the review text: 
        - Sentiment (positive or negative)
        - Is the reviewer expressing anger? (true or false)
        - Item purchased by reviewer
        - Company that made the item
        
        The review is delimited with triple backticks. \
        Format your response as a JSON object with \
        "Sentiment", "Anger", "Item" and "Brand" as the keys.
        If the information isn't present, use "unknown" \
        as the value.
        Make your response as short as possible.
        Format the Anger value as a boolean.
        
        Review text: '''{lamp_review}'''
        """
        ```
        
        提取主题
        
        ```python
        prompt = f"""
        Determine five topics that are being discussed in the \
        following text, which is delimited by triple backticks.
        
        Make each item one or two words long. 
        
        Format your response as a list of items separated by commas.
        
        Text sample: '''{story}'''
        """
        ```
        
        index主题，zero-shot
        
        ```python
        prompt = f"""
        Determine whether each item in the following list of \
        topics is a topic in the text below, which
        is delimited with triple backticks.
        
        Give your answer as list with 0 or 1 for each topic.\
        
        List of topics: {", ".join(topic_list)}
        
        Text sample: '''{story}'''
        """
        ```
        
        gpt让人兴奋的是，像上面提到的这些能力，在以前一个机器学习工程师可能要几天才能完成，而现在普通开发者通过api+prompt就可以实现。
        
        ---
        
        ## **Transforming**
        
        In this notebook, we will explore how to use Large Language Models for text transformation tasks such as language translation, spelling and grammar checking, tone adjustment, and format conversion.
        
        ****Translation****
        
        ****Tone Transformation****
        
        ```python
        prompt = f"""
        Translate the following from slang to a business letter: 
        'Dude, This is Joe, check out this spec on this standing lamp.'
        """
        ```
        
        ****Format Conversion，这个应该蛮常用的****
        
        ```python
        prompt = f"""
        Translate the following python dictionary from JSON to an HTML \
        table with column headers and title: {data_json}
        """
        ```
        
        ****Spellcheck/Grammar check.****
        
        ```python
        prompt = f"""Proofread and correct the following text
        and rewrite the corrected version. If you don't find
        and errors, just say "No errors found". Don't use 
        any punctuation around the text:
        ```{t}```"""
        ```
        
        ## **Expanding**
        
        In this lesson, you will generate customer service emails that are tailored to each customer's review.
        
        拓展，应用很多
        
        ```python
        prompt = f"""
        You are a customer service AI assistant.
        Your task is to send an email reply to a valued customer.
        Given the customer email delimited by ```, \
        Generate a reply to thank the customer for their review.
        If the sentiment is positive or neutral, thank them for \
        their review.
        If the sentiment is negative, apologize and suggest that \
        they can reach out to customer service. 
        Make sure to use specific details from the review.
        Write in a concise and professional tone.
        Sign the email as `AI customer agent`.
        Customer review: ```{review}```
        Review sentiment: {sentiment}
        """
        ```
        
         temperature
        
        越大，越不可预测，可能更有创意
        
        对于可预测应用，使用0，总是会选择可能最高，可以认为，相同输入，会有基本相同的输出
        
        同一个app，根据不同功能设置temperature，聊天可以高一点，总结类低
        
        ## **The Chat Format**
        
        In this notebook, you will explore how you can utilize the chat format to have extended conversations with chatbots personalized or specialized for specific tasks or behaviors.
        
        其实上下文，背景信息，就是通过assitant和system里提供
        
        ```python
        context = [ {'role':'system', 'content':"""
        You are OrderBot, an automated service to collect orders for a pizza restaurant. \
        You first greet the customer, then collects the order, \
        and then asks if it's a pickup or delivery. \
        You wait to collect the entire order, then summarize it and check for a final \
        time if the customer wants to add anything else. \
        If it's a delivery, you ask for an address. \
        Finally you collect the payment.\
        Make sure to clarify all options, extras and sizes to uniquely \
        identify the item from the menu.\
        You respond in a short, very conversational friendly style. \
        The menu includes \
        pepperoni pizza  12.95, 10.00, 7.00 \
        cheese pizza   10.95, 9.25, 6.50 \
        eggplant pizza   11.95, 9.75, 6.75 \
        fries 4.50, 3.50 \
        greek salad 7.25 \
        Toppings: \
        extra cheese 2.00, \
        mushrooms 1.50 \
        sausage 3.00 \
        canadian bacon 3.50 \
        AI sauce 1.50 \
        peppers 1.00 \
        Drinks: \
        coke 3.00, 2.00, 1.00 \
        sprite 3.00, 2.00, 1.00 \
        bottled water 5.00 \
        """} ]  # accumulate messages
        ```
        
        ---
        
        ## conclusion
        
        构建对人类有价值的应用。