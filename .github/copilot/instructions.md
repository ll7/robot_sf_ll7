# Copilot Instructions

You are assisting in a development workflow that uses GitHub Issues.

## General Behavior
- Always follow the latest coding standards and best practices.
- Use clear, descriptive variable and function names.
- Write code that is easy to read and maintain.
- Ensure that all code is well-documented with comments and docstrings.
- Write unit tests for new features and bug fixes.
- Use version control best practices, including meaningful commit messages.
- When making changes, ensure that they are backward compatible unless otherwise specified.
- Always check for existing issues or discussions before starting new work.
- Use the issue number in commit messages to link changes to specific issues.
- When working on a feature or bug fix, create a new branch named after the issue number and title in kebab-case.
  - Example: `feature/123-add-login-form`
- When pushing changes, ensure that the branch is up-to-date with the main branch to avoid merge conflicts.
- Use pull requests for code reviews and discussions before merging changes into the main branch.
- Always run tests before merging changes to ensure stability.
- Use descriptive commit messages that explain the purpose of the changes.

## Progress Reporting

- When generating progress reports, save them to the `progress/` directory.
- Each issue has its own subfolder under `progress/`, named after the issue number and title in kebab-case.
  - Example: `progress/123-add-login-form/`
- The report should be a markdown file named `progress-report.md`.
- Append a timestamp at the top of each new report.
- If the folder does not exist, create it.

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

