---
name: 🐛 Bug Report
about: Report a bug to help improve the project
title: "[BUG] "
labels: ["bug"]
assignees: []
---

## 🐛 Problem Description

**Brief summary:**
<!-- Provide a clear and concise description of the bug -->

**Expected behavior:**
<!-- Describe what you expected to happen -->

**Actual behavior:**
<!-- Describe what actually happened -->

**Impact:**
<!-- How does this affect users or the system? -->

## 🔄 Steps to Reproduce

1. <!-- First step -->
2. <!-- Second step -->
3. <!-- Additional steps -->
4. <!-- See error -->

**Minimal code example:**
```python
# Provide minimal code that reproduces the issue
```

## ✅ Acceptance Criteria

**Definition of Done:**
- [ ] Bug is fixed and no longer reproducible
- [ ] Root cause is identified and addressed
- [ ] Regression tests added to prevent future occurrences
- [ ] No new bugs introduced by the fix
- [ ] Existing functionality remains intact

**Testing Requirements:**
- [ ] Add test case that reproduces the bug
- [ ] Verify test fails before fix and passes after fix
- [ ] Run full test suite to ensure no regressions
- [ ] Test edge cases related to the bug

## 📁 Files and Components

**Files likely involved:**
- `path/to/suspected/file.py` - <!-- Why you suspect this file -->
- `path/to/another/file.py` - <!-- Additional suspected files -->

**System components affected:**
- [ ] Environment simulation
- [ ] Robot behavior
- [ ] Sensor systems
- [ ] Configuration handling
- [ ] Visualization
- [ ] Other: <!-- specify -->

**Test files to update:**
- `tests/test_*.py` - <!-- Specific test files to add cases -->

## 🔧 Error Information

**Error message:**
```
Paste the complete error message here
```

**Stack trace:**
```
Paste the complete stack trace here
```

**Console output:**
```
Paste relevant console output here
```

## 🌍 Environment

**System information:**
- OS: <!-- e.g., Ubuntu 22.04 -->
- Python version: <!-- e.g., 3.12.0 -->
- Robot SF version: <!-- git commit hash or tag -->

**Dependencies:**
<!-- Run: uv pip list | grep -E "(robot-sf|pysf|gymnasium)" -->
```
Paste relevant dependency versions here
```

**Configuration:**
<!-- Include relevant configuration that might affect the bug -->
```python
# Paste relevant configuration
```

## 🔍 Investigation Notes

**What I've tried:**
- [ ] <!-- Steps you've already attempted -->
- [ ] <!-- Additional debugging efforts -->

**Potential root cause:**
<!-- Share any insights about what might be causing the issue -->

**Workaround:**
<!-- If you found a temporary workaround, describe it here -->

## 📚 Related Information

**Related issues:**
- Related to # <!-- Link to related issues -->
- Possibly duplicates # <!-- Link if this might be a duplicate -->

**Documentation references:**
- <!-- Link to relevant documentation -->

**Recent changes:**
<!-- Any recent changes that might have introduced this bug -->

---

<!-- 
📝 **For GitHub Copilot:** This bug report follows the repository's development workflow.
Check the `.github/copilot/instructions.md` file for coding standards and practices.
Focus on creating focused, minimal fixes that address the root cause.
-->