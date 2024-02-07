import timeit
from transformers import BartTokenizer, BartForConditionalGeneration

# Initialize BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# Provided context
documents = ['The implementation of a quality management system (QMS) brings numerous benefits to organizations. A QMS is designed to ensure that processes and products consistently meet defined standards and customer expectations. Some key benefits of implementing a QMS include', '', 'Improved Product Quality A well-implemented QMS helps in consistently delivering high-quality products or services, leading to increased customer satisfaction', '', 'Enhanced Efficiency Streamlined processes and a focus on continuous improvement result in increased operational efficiency and reduced waste', '', "Compliance and Certification A QMS facilitates compliance with industry standards and regulations Obtaining certifications such as ISO 9001 can enhance the organization's reputation", '', 'Better Decision-Making Access to reliable data and metrics enables informed decision-making at all levels of the organization', '', 'Customer Satisfaction Meeting or exceeding customer expectations through consistent quality contributes to long-term customer loyalty', '', 'Risk Management Identifying and addressing potential risks in processes helps in preventing issues and disruptions', '', 'Employee Engagement Involving employees in the QMS fosters a culture of quality, empowering them to contribute to continuous improvement', '', 'Implementing a QMS requires commitment from leadership, employee involvement, and a focus on continuous learning and improvement',
             'The background check policy went into effect on 25 july 2001']

# Concatenate the documents into a single string
context = " ".join(documents)

# Tokenize the context
inputs = tokenizer(context, return_tensors="pt")

# Generate answers
start = timeit.default_timer()

generated_answers_encoded = model.generate(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["attention_mask"],
                                            min_length=64,
                                            max_length=256,
                                            do_sample=False, 
                                            early_stopping=True,
                                            num_beams=8,
                                            temperature=1.0,
                                            top_k=None,
                                            top_p=None,
                                            eos_token_id=tokenizer.eos_token_id,
                                            no_repeat_ngram_size=3,
                                            num_return_sequences=1)

# Decode generated answers and print
generated_answers = tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(generated_answers)

end = timeit.default_timer()
print(f'Total Time taken : {end-start}')


