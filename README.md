# GAN Stability Experiments

Проект содержит эксперименты по устойчивости обучения GAN.

## Эксперименты
- 2D toy-GAN
- MNIST (уменьшенный)

## Методы сравнения
- Vanilla GAN
- R1 regularization
- Instance Noise
- TTUR
- Extragradient / Optimistic GD

## Установка
pip install -r requirements.txt

## Запуск экспериментов
python mnist_experiment.py
python toy_2d_experiment.py

## Воспроизводимость
Все эксперименты используют фиксированные random seed.
