# ASK-RAG Interface Setup Guide  

## Step 1: Install Streamlit  

Make sure you have Python installed. Then install Streamlit globally or inside your virtual environment:  
``` bash  
pip install streamlit
```

## Create venv folder 
``` bash
python -m venv venv
```

# Activate the virtual environment
For windows :
``` bash
venv\Scripts\activate  
```
For macOS/Linux :  
``` bash
source venv/bin/activate  
```
# Install the required packages
``` bash
pip install -r requirements.txt  
```
# Run the main.py file  
``` bash
streamlit run main.py  
```
# Here is the final model RAG usecases  
The outlook of RAG model   
![]("https://github.com/parvesh-rana/rag_/blob/main/images/Screenshot%202025-08-12%20075727.png")  

Here you have to insert groq key i.e = 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   
![]("https://github.com/parvesh-rana/rag_/blob/main/images/Screenshot%202025-08-12%20075759.png")  

Then upload the document by clicking Browse files:   
![]("https://github.com/parvesh-rana/rag_/blob/main/images/Screenshot%202025-08-12%20080625.png")  

Now by clicking on Build/Rebuild index generate vecotor index of embeddings.   
![]("https://github.com/parvesh-rana/rag_/blob/main/images/Screenshot%202025-08-12%20080725.png")  

Ask you question:  
![]("https://github.com/parvesh-rana/rag_/blob/main/images/Screenshot%202025-08-12%20081307.png")  
![]("https://github.com/parvesh-rana/rag_/blob/main/images/Screenshot%202025-08-12%20084429.png")  
![]("https://github.com/parvesh-rana/rag_/blob/main/images/Screenshot%202025-08-12%20084448.png")
