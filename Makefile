# Makefile untuk StressGuard Web App
# Kompatibel dengan mingw32-make (Windows) dan GNU make (Linux/macOS/Git Bash)

ifeq ($(OS),Windows_NT)
	ifneq ($(wildcard .venv/Scripts/python.exe),)
		PYTHON := .venv\Scripts\python.exe
	else ifneq ($(wildcard $(subst \,/,$(LOCALAPPDATA))/Python/bin/python.exe),)
		PYTHON := "$(LOCALAPPDATA)\Python\bin\python.exe"
	else
		PYTHON := python
	endif
else
	ifneq ($(wildcard .venv/bin/python),)
		PYTHON := .venv/bin/python
	else
		PYTHON := python3
	endif
endif

PORT ?= 8000

.PHONY: help run train test check install clean

help:
	@echo ============================================================
	@echo   StressGuard - Perintah Makefile Tersedia
	@echo ============================================================
	@echo   make run      : Jalankan server web di port $(PORT)
	@echo   make test     : Jalankan pengujian otomatis (unit test)
	@echo   make check    : Periksa kesehatan konfigurasi Django
	@echo   make train    : Latih ulang model Random Forest (.joblib)
	@echo   make install  : Pasang library dari requirements.txt
	@echo   make clean    : Bersihkan folder cache __pycache__
	@echo ============================================================

run:
	$(PYTHON) manage.py runserver 127.0.0.1:$(PORT)

test:
	$(PYTHON) manage.py test prediction -v 2

check:
	$(PYTHON) manage.py check

train:
	$(PYTHON) train_model.py

install:
	$(PYTHON) -m pip install -r requirements.txt

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
