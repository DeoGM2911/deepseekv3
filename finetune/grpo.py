from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward


def train(dataset_name, split, model_name, reward_func):
    dataset = load_dataset(dataset_name, split=split)
    trainer = GRPOTrainer(
        model_name=model_name,
        dataset=dataset,
        reward_func=reward_func,
        config=GRPOConfig(
            num_epochs=1,
            learning_rate=1e-5,
            weight_decay=0.01,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            greater_is_better=True,
        )
    )
    
