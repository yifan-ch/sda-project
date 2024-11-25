# Makefile

# Default target: only install
all: install

# Compile the requirements, and install the missing ones
dev: compile install

# Install dependencies from requirements.txt
install:
	pip install -r requirements.txt

# Generate requirements.in using pipreqs
requirements.in:
	pipreqs . --savepath=requirements.in

# Compile requirements.txt from requirements.in
compile: requirements.in
	pip-compile requirements.in

# Clean generated files
clean:
	rm -f requirements.in requirements.txt

# Run everything from scratch
rebuild: clean all

.PHONY: all install clean rebuild compile
