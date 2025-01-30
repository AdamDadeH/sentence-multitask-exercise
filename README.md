Please see the attached `discussion.pdf` and/or `discussion.md` for more details : 

Pretraining and embedding examples are run via : 
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

cd src
python3 pretrain.py
python3 test_embeddings.py
python3 multitask_test.py
```

Generating discussion pdf : 
```bash
pandoc discussion.md -o discussion.pdf
```
