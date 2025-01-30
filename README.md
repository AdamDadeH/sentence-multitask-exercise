Please see the attached `discussion.pdf` and/or `discussion.md` for more details : 

Pretraining and embedding examples are run via : 
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

python3 src/pretrain.py

python3 test_embeddings.py
```

Generating discussion pdf : 
```bash
pandoc discussion.md -o discussion.pdf
```
