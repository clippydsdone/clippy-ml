

# Clippy ML
This repository contains resources to host your own summarization model on Azure. 

The code in the `azure-example` folder contains Microsofts' Azure [Guide for hosting BART summarization model with Azure ML Studio](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-nlp-processing-batch?tabs=cli). 

Code in jupyter notebook [new_model.ipynb](https://github.com/clippydsdone/clippy-ml/blob/main/new_model.ipynb "new_model.ipynb") contains minimal code needed to host BART on your own Azure instance with API inference route. The idea is that BART is hosted and with an HTTP request you get a response with summarized text.

Beware that you will need a very good **GPU** instance to host pretrained BART. 

Good luck!

## Resources
- [HuggingFace BART page](https://huggingface.co/facebook/bart-large-cnn)
- [Azure ML Studio Resources](https://learn.microsoft.com/en-us/azure/machine-learning/)
- [NLP Cloud Summarization Guidlines](https://docs.nlpcloud.com/#summarization)


