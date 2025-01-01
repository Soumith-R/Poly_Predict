import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def log_training_info(epoch, loss):
    logging.info(f'Epoch: {epoch}, Loss: {loss:.4f}')