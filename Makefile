# Makefile

# Default target
all: clean compile install

# run if on windows
win: clean-win compile install

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

clean-win:
	del requirements.in



.PHONY: all install compile clean
