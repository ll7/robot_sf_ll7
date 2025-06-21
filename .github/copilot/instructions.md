# Copilot Instructions

You are assisting in a development workflow that uses GitHub Issues and follows software engineering best practices.

## General Behavior

### Code Quality Standards
- Always follow the latest coding standards and best practices for the language being used
- Use clear, descriptive variable and function names that express intent
- Write code that is easy to read, understand, and maintain
- Ensure that all code is well-documented with meaningful comments and docstrings
- Follow the existing code style and patterns in the project
- Write comprehensive unit tests for new features and bug fixes
- Perform code reviews and ensure changes meet quality standards

### Version Control & Collaboration
- Use version control best practices with meaningful, descriptive commit messages
- When making changes, ensure backward compatibility unless explicitly specified otherwise
- Always check for existing issues, discussions, or similar work before starting new tasks
- Use issue numbers in commit messages to link changes to specific GitHub issues
  - Format: `fix: resolve button alignment issue (#42)`
- Create feature branches named after the issue number and title in kebab-case
  - Format: `feature/42-fix-button-alignment` or `bugfix/123-memory-leak-fix`
- Keep branches up-to-date with the main branch to avoid merge conflicts
- Use pull requests for code reviews and team discussions before merging
- Always run the full test suite before merging changes to ensure system stability

### Problem-Solving Approach
- Break down complex problems into smaller, manageable tasks
- Research existing solutions and patterns before implementing new approaches
- Consider the impact of changes on the entire system, not just the immediate problem
- Document architectural decisions and trade-offs made during implementation
- Think about edge cases, error handling, and potential failure modes

## Documentation Standards

### Technical Documentation
- Create comprehensive documentation for all significant changes and new features
- Save documentation files in the `docs/` directory using a clear folder structure
- Each major feature or issue should have its own subfolder named in kebab-case
  - Format: `docs/42-fix-button-alignment/` or `docs/feature-name/`
- Use descriptive README.md files as the main documentation entry point for each folder

### Documentation Content Requirements
Documentation should include:
- **Problem Statement**: Clear description of the issue being addressed
- **Solution Overview**: High-level approach and architectural decisions
- **Implementation Details**: Code examples, API changes, and technical specifics
- **Impact Analysis**: What systems/users are affected and how
- **Testing Strategy**: How the changes were validated
- **Future Considerations**: Potential improvements or known limitations
- **Related Links**: References to GitHub issues, pull requests, or external resources

### Documentation Best Practices
- Use proper markdown formatting with clear headings and structure
- Include code examples with syntax highlighting
- Add diagrams or screenshots when they improve understanding
- Write for future developers who may be unfamiliar with the context
- Keep documentation up-to-date as code evolves
- Use consistent formatting and follow markdown linting standards

## Project-Specific Guidelines

### Robot SF Development
- This project focuses on robotic simulation and reinforcement learning
- Pay special attention to data integrity in simulation states and analysis
- Ensure consistency between simulation data generation and analysis pipelines
- Consider the impact on research workflows and data analysis tools
- Maintain compatibility with the fast-pysf reference implementation when applicable
- Test changes thoroughly as they may affect both simulation behavior and research results

## Examples

### Branch Naming
```
feature/42-fix-button-alignment
bugfix/89-memory-leak-in-simulator
enhancement/156-improve-lidar-performance
```

### Commit Messages
```
fix: resolve 2x speed multiplier in VisualizableSimState (#42)
feat: add new lidar sensor configuration options (#156)
docs: update installation guide with GPU setup instructions
test: add comprehensive integration tests for pedestrian simulation
```

### Documentation Structure
```
docs/42-fix-button-alignment/
├── README.md              # Main documentation
├── before-after-comparison.md
└── implementation-notes.md
```lopment workflow that uses GitHub Issues and follows software engineering best practices.

## General Behavior

### Code Quality Standards
- Always follow the latest coding standards and best practices for the language being used
- Use clear, descriptive variable and function names that express intent
- Write code that is easy to read, understand, and maintain
- Ensure that all code is well-documented with meaningful comments and docstrings
- Follow the existing code style and patterns in the project
- Write comprehensive unit tests for new features and bug fixes
- Perform code reviews and ensure changes meet quality standards

### Version Control & Collaboration
- Use version control best practices with meaningful, descriptive commit messages
- When making changes, ensure backward compatibility unless explicitly specified otherwise
- Always check for existing issues, discussions, or similar work before starting new tasks
- Use issue numbers in commit messages to link changes to specific GitHub issues
  - Format: `fix: resolve button alignment issue (#42)`
- Create feature branches named after the issue number and title in kebab-case
  - Format: `feature/42-fix-button-alignment` or `bugfix/123-memory-leak-fix`
- Keep branches up-to-date with the main branch to avoid merge conflicts
- Use pull requests for code reviews and team discussions before merging
- Always run the full test suite before merging changes to ensure system stability

### Problem-Solving Approach
- Break down complex problems into smaller, manageable tasks
- Research existing solutions and patterns before implementing new approaches
- Consider the impact of changes on the entire system, not just the immediate problem
- Document architectural decisions and trade-offs made during implementation
- Think about edge cases, error handling, and potential failure modes

## Markdown Documentation

- Use markdown files for documentation.
- Save documentation files in the `docs/` directory.
- Each issue has its own subfolder under `docs/`, named after the issue number and title in kebab-case.
  - Example: `docs/123-add-login-form/`
- The documentation should include:
  - A summary of the issue and its resolution
  - Code examples where applicable
  - Links to related issues or pull requests
  - Any relevant diagrams or images
- Use headings, lists, and code blocks to format the documentation for clarity.
- Write markdownlint to ensure the documentation is well-formatted and adheres to markdown standards.

## Example

If we are working on issue #42 titled “Fix button alignment”, then save the file to:

