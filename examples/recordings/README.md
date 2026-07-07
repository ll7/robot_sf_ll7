# Robot SF Recordings

Pickle recordings shared in this directory are accepted only through a
restricted unpickler that allowlists the exact RobotSF classes and NumPy
reconstruction symbols present in known-good recordings. Arbitrary pickle
files remain untrusted and will be rejected with an `UnsafePickleError` if
they reference symbols outside the allowlist. Prefer JSONL or other safer
formats for new recordings; format migration from pickle is post-submission
work.
