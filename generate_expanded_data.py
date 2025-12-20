#!/usr/bin/env python3
"""Generate expanded workout dataset with 138 samples per class."""

import pandas as pd
import numpy as np

# Base patterns for each exercise (mean values)
patterns = {
    'pushup': {'Ax': -9.5, 'Ay': 2.5, 'Az': 15.8, 'Gx': 0.5, 'Gy': 0.3, 'Gz': -1.2},
    'squat': {'Ax': -1.2, 'Ay': 8.5, 'Az': 9.8, 'Gx': -0.5, 'Gy': 2.1, 'Gz': 0.3},
    'jumping_jack': {'Ax': 0.2, 'Ay': 0.1, 'Az': 9.2, 'Gx': 1.2, 'Gy': -1.5, 'Gz': -2.1},
    'bench_press': {'Ax': -11.5, 'Ay': 3.2, 'Az': 18.1, 'Gx': 0.8, 'Gy': 0.6, 'Gz': -0.8},
    'bicepcurls': {'Ax': -11.8, 'Ay': 6.0, 'Az': 19.4, 'Gx': 1.7, 'Gy': 1.5, 'Gz': -2.5},
    'deadlift': {'Ax': -5.2, 'Ay': 1.8, 'Az': 12.3, 'Gx': -1.2, 'Gy': 3.5, 'Gz': 1.2},
    'lunges': {'Ax': -7.1, 'Ay': 5.2, 'Az': 14.6, 'Gx': 0.2, 'Gy': 1.8, 'Gz': -1.5},
    'plank': {'Ax': -2.5, 'Ay': 9.1, 'Az': 8.7, 'Gx': -0.8, 'Gy': 0.5, 'Gz': 0.9},
    'burpees': {'Ax': 1.5, 'Ay': 1.2, 'Az': 10.5, 'Gx': 1.8, 'Gy': -2.1, 'Gz': -0.5},
    'mountain_climbers': {'Ax': 3.2, 'Ay': -0.5, 'Az': 11.2, 'Gx': 2.1, 'Gy': -1.8, 'Gz': -1.2}
}

weights = {
    'pushup': 0, 'squat': 0, 'jumping_jack': 0, 'bench_press': 135,
    'bicepcurls': 15, 'deadlift': 185, 'lunges': 0, 'plank': 0,
    'burpees': 0, 'mountain_climbers': 0
}

np.random.seed(42)
rows = []
task_id = 30001  # Start from 30001 like your original data

SAMPLES_PER_CLASS = 138  # Match your original bicep curls data

for exercise, pattern in patterns.items():
    for i in range(SAMPLES_PER_CLASS):
        time = (i + 1) * 10  # 10ms intervals, up to 1380ms like your data
        
        # Add realistic variation with slight temporal trends
        progress = i / SAMPLES_PER_CLASS  # 0 to 1 across the movement
        
        # Noise scales
        noise_scale = 0.12
        
        # Add slight sinusoidal variation to simulate movement phases
        phase_var = np.sin(progress * 2 * np.pi) * 0.5
        
        row = {
            'TaskId': task_id,
            'ExerciseName': exercise,
            'Weight': weights[exercise],
            'Ax': pattern['Ax'] + np.random.normal(0, abs(pattern['Ax']) * noise_scale) + phase_var * 0.3,
            'Ay': pattern['Ay'] + np.random.normal(0, abs(pattern['Ay']) * noise_scale) + phase_var * 0.2,
            'Az': pattern['Az'] + np.random.normal(0, abs(pattern['Az']) * noise_scale) + phase_var * 0.4,
            'Gx': pattern['Gx'] + np.random.normal(0, max(abs(pattern['Gx']) * noise_scale, 0.15)) + phase_var * 0.5,
            'Gy': pattern['Gy'] + np.random.normal(0, max(abs(pattern['Gy']) * noise_scale, 0.15)) + phase_var * 0.4,
            'Gz': pattern['Gz'] + np.random.normal(0, max(abs(pattern['Gz']) * noise_scale, 0.15)) + phase_var * 0.3,
            'Time': time,
            'Phase': 'concentric'
        }
        rows.append(row)
    task_id += 1

df = pd.DataFrame(rows)
df.to_csv('data/workouts_multiclass.csv', index=False)

print('=' * 70)
print('Dataset Generation Complete!')
print('=' * 70)
print(f'Total samples: {len(df)}')
print(f'Classes: {len(patterns)}')
print(f'Samples per class: {SAMPLES_PER_CLASS}')
print(f'Time range per exercise: 10ms - {df.Time.max()}ms')
print(f'\nClass distribution:')
for exercise in sorted(df.ExerciseName.unique()):
    count = len(df[df.ExerciseName == exercise])
    print(f'  {exercise:20} : {count:3} samples')
print('=' * 70)
