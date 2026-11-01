PY=python

install:
	pip install -r requirements.txt
ingest:
	$(PY) -m civilrag ingest
ask:
	$(PY) -m civilrag ask "$(Q)"
chat:
	$(PY) -m civilrag chat
serve:
	$(PY) -m civilrag serve
eval:
	$(PY) -m civilrag eval
test:
	pytest -q
