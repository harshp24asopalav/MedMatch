# Define variables
VENV := venv
REQ := requirements.txt

.PHONY: all create_venv install_deps activate run clean

# Default target
all: create_venv install_deps

# Create virtual environment
create_venv:
	@echo "Creating virtual environment..."
	@python -m venv $(VENV)

# Install dependencies
install_deps: create_venv
	@echo "Installing dependencies..."
	@$(VENV)/Scripts/pip install -r $(REQ) || $(VENV)/bin/pip install -r $(REQ)

# Activate virtual environment
activate:
	@echo "Activating virtual environment..."
	@$(VENV)/Scripts/activate || source $(VENV)/bin/activate

# Run the application
run:
	@echo "Running the application..."
	@$(VENV)/Scripts/python app.py || $(VENV)/bin/python app.py

# Clean the virtual environment
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)