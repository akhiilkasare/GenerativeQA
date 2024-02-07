

'''https://huggingface.co/vblagoje/bart_lfqa'''

question = "What are the benefits of implementing a quality management system?"

context = '''The implementation of a quality management system (QMS) brings numerous benefits to organizations. A QMS is designed to ensure that processes and products consistently meet defined standards and customer expectations Some key benefits of implementing a QMS include

Improved Product Quality A well-implemented QMS helps in consistently delivering high-quality products or services, leading to increased customer satisfaction

Enhanced Efficiency Streamlined processes and a focus on continuous improvement result in increased operational efficiency and reduced waste

Compliance and Certification A QMS facilitates compliance with industry standards and regulations Obtaining certifications such as ISO 9001 can enhance the organization's reputation

Better Decision-Making Access to reliable data and metrics enables informed decision-making at all levels of the organization

Customer Satisfaction Meeting or exceeding customer expectations through consistent quality contributes to long-term customer loyalty

Risk Management Identifying and addressing potential risks in processes helps in preventing issues and disruptions

Employee Engagement Involving employees in the QMS fosters a culture of quality, empowering them to contribute to continuous improvement

Implementing a QMS requires commitment from leadership, employee involvement, and a focus on continuous learning and improvement
'''


import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import timeit

start = timeit.default_timer()

model_name = "bart_lfqa"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# model = model.to(device)

# it all starts with a question/query
query = "When did the background check policy went into effect?"

# given the question above suppose these documents below were found in some document store 
# documents = ["The implementation of a quality management system (QMS) brings numerous benefits to organizations.",
#              " A QMS is designed to ensure that processes and products consistently meet defined standards and customer expectations",
#              "some key benefits of implementing a QMS include:",
#              "Improved Product Quality: A well-implemented QMS helps in consistently delivering high-quality products or services, leading to increased customer satisfaction.",            
#              "Better Decision-Making: Access to reliable data and metrics enables informed decision-making at all levels of the organization.",
#              "Customer Satisfaction: Meeting or exceeding customer expectations through consistent quality contributes to long-term customer loyalty.,""when the skin is completely wet. The body continuously loses water by...",
#              "at greater pressures. There is an ambiguity, however, as to the meaning of the terms 'heating' and 'cooling'...",
#              "are not in a relation of thermal equilibrium, heat will flow from the hotter to the colder, by whatever pathway...",
#              "air condition and moving along a line of constant enthalpy toward a state of higher humidity. A simple example ...",            
#              "Thermal contact conductance In physics, thermal contact conductance is the study of heat conduction between solid ..."]

documents=['The implementation of a quality management system (QMS) brings numerous benefits to organizations. A QMS is designed to ensure that processes and products consistently meet defined standards and customer expectations Some key benefits of implementing a QMS include', '', 'Improved Product Quality A well-implemented QMS helps in consistently delivering high-quality products or services, leading to increased customer satisfaction', '', 'Enhanced Efficiency Streamlined processes and a focus on continuous improvement result in increased operational efficiency and reduced waste', '', "Compliance and Certification A QMS facilitates compliance with industry standards and regulations Obtaining certifications such as ISO 9001 can enhance the organization's reputation", '', 'Better Decision-Making Access to reliable data and metrics enables informed decision-making at all levels of the organization', '', 'Customer Satisfaction Meeting or exceeding customer expectations through consistent quality contributes to long-term customer loyalty', '', 'Risk Management Identifying and addressing potential risks in processes helps in preventing issues and disruptions', '', 'Employee Engagement Involving employees in the QMS fosters a culture of quality, empowering them to contribute to continuous improvement', '', 'Implementing a QMS requires commitment from leadership, employee involvement, and a focus on continuous learning and improvement',
'The background check policy went into effect on 25 july 2001']



# concatenate question and support documents into BART input
conditioned_doc = "<P> " + " <P> ".join([d for d in documents])
query_and_docs = "question: {} context: {}".format(query, conditioned_doc)

model_input = tokenizer(query_and_docs, truncation=True, padding=True, return_tensors="pt")

generated_answers_encoded = model.generate(input_ids=model_input["input_ids"],
                                          attention_mask=model_input["attention_mask"],
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
print(tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True,clean_up_tokenization_spaces=True))
end = timeit.default_timer()
print(f'Total Time taken : {end-start}')



