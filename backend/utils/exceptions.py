class CustomError(Exception):
    """Base class for custom exceptions in this module."""
    pass

class AuthenticationError(CustomError): pass

class SetupError(Exception): 
    pass

class ADAuthenticationError(SetupError): 
    pass

class NativeAuthenticationError(SetupError): 
    pass

class InvalidCredentialsError(CustomError):
    """Raised when user credentials are invalid."""
    pass

class MissingCredentialsError(CustomError):
    """Raised when username or password is not provided."""
    pass

class UserNotMappedError(CustomError):
    """Raised when user exists but isn't mapped to any profile."""
    pass

class EngineConnectionError(CustomError):
    """Raised when unable to create a database engine."""
    pass

class ConfigurationError(Exception):
    pass

class UserNotFoundError(Exception):
    pass

class UserNotFoundError(CustomError):
    """Raised when user is not found."""
    def __init__(self, message="User not found", user_id=None):
        super().__init__(message)
        self.user_id = user_id
    def __str__(self):
        return f"{self.args[0]} (User ID: {self.user_id})"
    
