# Contributing to the Neural Circuit Extraction Framework

We're excited that you're interested in contributing to the Neural Circuit Extraction Framework! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md) that all contributors are expected to follow. Please read it before participating.

## How to Contribute

### Reporting Bugs

If you find a bug, please report it by creating an issue on GitHub. When filing an issue, make sure to include:

- A clear, descriptive title
- A detailed description of the issue, including steps to reproduce
- Your environment information (OS, Python version, etc.)
- Any relevant logs or screenshots

### Suggesting Enhancements

We welcome suggestions for enhancements! To suggest an enhancement:

1. Create an issue on GitHub with the tag "enhancement"
2. Provide a clear description of the proposed functionality
3. Explain why this enhancement would be valuable
4. If possible, outline how the enhancement might be implemented

### Pull Requests

We actively welcome pull requests:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add or update tests as necessary
5. Update documentation as needed
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

#### Pull Request Guidelines

- Each pull request should focus on a single feature or bug fix
- Include tests for any new functionality
- Update documentation for any changed functionality
- Follow the project's code style and conventions
- Make sure all tests pass before submitting
- Include a clear and descriptive PR title and description

## Development Setup

To set up your development environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/simplest_arc_agi.git
   cd simplest_arc_agi
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Project Structure

- `src/` - Source code organized by component
- `tests/` - Test suite
- `docs/` - Documentation
- `examples/` - Example notebooks and scripts

## Documentation

We use MkDocs for documentation. When contributing:

- Document any new functionality or changes to existing functionality
- Update examples if they're affected by your changes
- To preview documentation changes:
  ```bash
  mkdocs serve
  ```

## Testing

- Write tests for new functionality
- Make sure existing tests pass with your changes
- Run the test suite:
  ```bash
  pytest
  ```

## Style Guide

We follow PEP 8 for code style. Additionally:

- Use descriptive variable names
- Write docstrings for all functions, classes, and modules
- Keep functions focused on a single responsibility
- Type annotations are encouraged

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Questions?

If you have questions about contributing, please open an issue or reach out to the maintainers.

Thank you for your contributions! 