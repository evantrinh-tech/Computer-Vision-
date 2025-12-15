import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from typing import Dict, Any, Optional
import pandas as pd
import yaml

from src.training.trainer import ModelTrainer
from src.data_processing.collectors import SimulatedDataCollector
from src.utils.config import settings
from src.utils.logger import logger

def run_training_pipeline(
    model_type: str = 'ANN',
    config_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    use_simulated_data: bool = False
) -> Dict[str, Any]:

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    trainer = ModelTrainer(model_type=model_type, config_path=config_path)

    if model_type == 'CNN':
        if data_path:
            data_path_obj = Path(data_path)
            logger.info(f"Đang load dữ liệu ảnh từ {data_path_obj}...")

            if data_path_obj.is_file():
                data_path_obj = data_path_obj.parent
                logger.info(f"Đã nhận file, sử dụng thư mục: {data_path_obj}")

            X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(
                data_path=data_path_obj
            )
        else:
            raise ValueError("CNN model cần data_path đến thư mục chứa ảnh")
    else:
        if use_simulated_data:
            logger.info("Sử dụng dữ liệu mô phỏng...")
            collector = SimulatedDataCollector()
            df_normal = collector.generate_sensor_data(n_samples=1000, has_incident=False)
            df_incident = collector.generate_sensor_data(n_samples=200, has_incident=True)
            df = pd.concat([df_normal, df_incident], ignore_index=True)
        elif data_path:
            logger.info(f"Đang load dữ liệu từ {data_path}...")
            df = pd.read_csv(data_path)
        else:
            raise ValueError("Cần cung cấp data_path hoặc set use_simulated_data=True")

        X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(df=df)

    with mlflow.start_run(run_name=f"{model_type}_training"):
        if model_type == 'CNN':
            mlflow.log_param("n_samples", len(X_train) + len(X_val) + len(X_test))
            mlflow.log_param("image_shape", X_train.shape[1:])
        else:
            mlflow.log_param("n_samples", len(df))
            mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))

        training_results = trainer.train(X_train, y_train, X_val, y_val)

        test_metrics = trainer.evaluate_on_test(X_test, y_test)

        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        logger.info("Training pipeline completed successfully!")

        return {
            'training_results': training_results,
            'test_metrics': test_metrics,
            'model_path': training_results.get('model_path')
        }

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run training pipeline')
    parser.add_argument('--model', type=str, default='ANN', choices=['ANN', 'RBFNN', 'CNN', 'RNN'])
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--simulate', action='store_true', help='Use simulated data')

    args = parser.parse_args()

    run_training_pipeline(
        model_type=args.model,
        config_path=Path(args.config) if args.config else None,
        data_path=Path(args.data) if args.data else None,
        use_simulated_data=args.simulate
    )