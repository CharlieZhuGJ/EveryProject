from main import logger


def record_log(info="Running training", train_examples=None, train_batch_size=0,
               num_train_optimization_steps=0):
    if train_examples is None:
        train_examples = []
    logger.info("****** %s  ******", info)
    logger.info(" Num examples = %d", len(train_examples))
    logger.info(" Batch size = %d", train_batch_size)
    logger.info(" Num steps = %d", num_train_optimization_steps)
