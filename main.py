import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Create a simple model
model = Sequential()
model.add(Dense(18, input_dim=9, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=0.01))

# Example training loop
def train_model(model, epochs=1000):
    for epoch in range(epochs):
        # Generate a game state and corresponding move
        state = np.random.choice([0, 1, -1], size=(9,))
        next_move = np.random.choice(range(9))
        
        # Simulate reward (in reality, you'd play the game)
        reward = np.random.choice([1, -1, 0])
        
        # Predict and update model
        target = reward + 0.99 * np.max(model.predict(state.reshape(1, -1)))
        target_f = model.predict(state.reshape(1, -1))
        target_f[0][next_move] = target
        
        model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

# Train the model
train_model(model)
