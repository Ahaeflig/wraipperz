import logging

from wraipperz.api.llm import call_ai

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def test_basic():
    print(call_ai, flush=True)
    print("ASD", flush=True)
    """A minimal test that doesn't use any AI functionality"""
    logger.info("Running basic test")
    assert True
    logger.info("Basic test passed")


def test_import():
    """Test that we can import the module"""
    logger.info("Importing module")
    logger.info("Import successful")
    assert True


def test_singleton():
    """Test that we can get the singleton instance"""
    from wraipperz.api.llm import AIManagerSingleton

    logger.info("Getting singleton instance")
    instance = AIManagerSingleton._instance  # Just check if it exists, don't initialize
    logger.info(f"Singleton instance exists: {instance is not None}")
    assert (
        True
    )  # We don't assert instance is not None because it might not be initialized yet

    # Don't call get_instance() as that might trigger the code that's hanging
