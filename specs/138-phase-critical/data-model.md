# Data Model: Type Checking Entities

## TypeAnnotation Entity
- **Fields**:
  - name: str (annotation identifier)
  - target: str (function/class/variable being annotated)
  - type_hint: str (the type annotation string)
  - module: str (file path)
  - line: int (line number)
- **Validation Rules**:
  - Must be valid Python type syntax
  - Must match runtime behavior
  - Must be importable in context
- **Relationships**:
  - Belongs to Module
  - References Type entities

## FactoryFunction Entity
- **Fields**:
  - name: str (function name)
  - parameters: List[Parameter] (input parameters)
  - return_type: TypeAnnotation (output type)
  - module: str
  - decorators: List[str]
- **Validation Rules**:
  - Must return gymnasium.Env instance
  - Must accept config parameter
  - Must be callable without side effects
- **Relationships**:
  - Creates Environment instances
  - Uses Configuration

## ImportStatement Entity
- **Fields**:
  - module: str (imported module)
  - alias: Optional[str] (import alias)
  - is_conditional: bool (TYPE_CHECKING guard)
  - required: bool (mandatory vs optional)
- **Validation Rules**:
  - Conditional imports must use TYPE_CHECKING
  - Optional imports must handle ImportError
  - All imports must be resolvable
- **Relationships**:
  - Used by Modules
  - References external packages

## TypeDiagnostic Entity
- **Fields**:
  - file: str (affected file)
  - line: int (line number)
  - column: int (column number)
  - message: str (error description)
  - category: str (invalid-arg, unresolved-import, etc.)
  - severity: str (error/warning)
- **Validation Rules**:
  - Must have valid file path
  - Must have actionable message
  - Must be reproducible
- **Relationships**:
  - Belongs to Module
  - Resolved by TypeFix

## State Transitions
- **TypeDiagnostic Lifecycle**:
  - Detected → Categorized → Prioritized → Fixed → Validated
- **ImportStatement Lifecycle**:
  - Required → Optional (if not critical) → Conditional → Resolved