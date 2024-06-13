# Define variables
VENV := venv
UNAME_S := $(shell python -c "import platform; print(platform.system())")

# Determine the Python and pip executable paths based on the OS
ifeq ($(UNAME_S),Linux)
	PYTHON := $(VENV)/bin/python
	PIP := $(VENV)/bin/pip
else ifeq ($(UNAME_S),Darwin)
	PYTHON := $(VENV)/bin/python
	PIP := $(VENV)/bin/pip
else
	PYTHON := $(VENV)/Scripts/python
	PIP := $(VENV)/Scripts/pip
endif

# Targets

.PHONY: all setup run clean

all: setup run

setup:
	@echo "Setting up virtual environment..."
	python -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

run:
	@echo "Running Flask server..."
	$(PYTHON) app.py

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +

