Feature: Episode Schema Validation
  As a developer running benchmark episodes
  I want to validate episode records against the canonical schema
  So that malformed data is caught early

  Scenario: Valid episode record passes schema validation
    Given a valid minimal episode record
    When the record is validated against the episode schema
    Then no validation error should be raised

  Scenario: Malformed episode record is rejected
    Given an episode record missing a required field
    When the record is validated against the episode schema
    Then a validation error should be raised
