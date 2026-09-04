# Contributing to MetaPulsar

We welcome contributions to MetaPulsar! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://www.github.com/vhaasteren/metapulsar.git
   cd metapulsar
   ```

2. **Install in Development Mode** (Python 3.11+; pulls git PINT `@metapulsar` and git nltiming)
   ```bash
   pip install -e ".[dev,libstempo]"
   ```

   Prefer an editable [nanograv/PINT](https://github.com/nanograv/PINT/tree/main)
   checkout when developing against PINT:

   ```bash
   pip install -e . --no-deps   # from the local PINT checkout
   ```

   A stale non-editable `pint` under `~/.local/lib/.../site-packages` can shadow
   that install: user site precedes the editable finder on `sys.path`, and
   `pip show` still reports the *correct version from the wrong metadata*.
   Diagnose with `pathlib.Path(pint.__file__).resolve()`, never with `pip show`
   alone.

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   pytest
   ```

## 📝 Development Guidelines

### Code Style

- **Python**: Follow PEP 8 style guidelines
- **Formatting**: Use `black` for code formatting
- **Linting**: Use `ruff` for linting
- **Type Hints**: Use type hints for all functions and methods
- **Docstrings**: Follow Google-style docstrings

### Testing

- **Coverage**: Maintain high test coverage (>90%)
- **Test Categories**: Use appropriate pytest markers
  - `@pytest.mark.slow` for slow tests
  - `@pytest.mark.integration` for integration tests
- **Mock Timing Objects**: Use `create_mock_libstempo` / `MockLibstempo` for timing-mock based tests

### Documentation

- **Docstrings**: All public functions must have docstrings
- **Examples**: Include usage examples in docstrings
- **Type Hints**: Use type hints for better IDE support

## 🔄 Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements
- `refactor/description` - Code refactoring

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Changes**
   ```bash
   pytest
   black --check src/ tests/
   ruff check src/ tests/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🧪 Testing Guidelines

### Test Structure

```python
class TestFeatureName:
    """Test class for FeatureName functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        pass
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        pass
```

### MockLibstempo Usage

Use `MockLibstempo` for testing timing-object integration:

```python
from metapulsar.mockpulsar import create_mock_libstempo

def test_metapulsar_creation(mock_metapulsar):
    """Test MetaPulsar creation with MockLibstempo.

    MetaPulsar requires `pta_files` for every PTA -- it reads each leg's par
    text from that file and never re-serializes an engine object. The
    `mock_metapulsar` fixture (tests/conftest.py) writes the mocks' own pars
    under tmp_path and passes them; `mockpulsar.write_mock_pta_files()` does the
    same outside a pytest fixture.
    """
    mock_lt = create_mock_libstempo(
        n_toas=100, name="J1857+0943", telescope="test", seed=42
    )
    metapulsar = mock_metapulsar({"test": mock_lt})
    assert len(metapulsar._toas) == 100
```

## 📚 Documentation Guidelines

### Docstring Format

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
        RuntimeError: When operation fails
        
    Example:
        >>> result = function_name("test", 20)
        >>> print(result)
        True
    """
```

### API Documentation

- All public classes and functions must be documented
- Include usage examples in docstrings
- Document parameters, return values, and exceptions
- Use type hints consistently

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal code example
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, package versions
6. **Error Messages**: Full traceback if applicable

## ✨ Feature Requests

When requesting features, please include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Additional Context**: Any other relevant information

## 📋 Code Review Checklist

### For Contributors

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Type hints are used
- [ ] No debugging code left behind
- [ ] Commit messages are clear

### For Reviewers

- [ ] Code is readable and well-structured
- [ ] Tests cover the new functionality
- [ ] Documentation is clear and complete
- [ ] No breaking changes without justification
- [ ] Performance implications considered

## 🤝 Community Guidelines

- Be respectful and constructive
- Help others learn and improve
- Ask questions when unsure
- Share knowledge and best practices
- Follow the code of conduct

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://www.github.com/vhaasteren/metapulsar/issues)
- **Email**: [rutger@vhaasteren.com](mailto:rutger@vhaasteren.com)

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.
