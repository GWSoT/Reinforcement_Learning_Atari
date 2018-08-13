import keras
from keras.layers import Dense, Input, Conv2D, Flatten, Lambda, Add
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras import backend as K
K.set_image_dim_ordering('th')


def create_dqn_network(input_shape, output_shape, optimizer='adam', loss='mse'):
    model = Sequential()

    model.add(Dense(24, input_dim=(1, 4), activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(2, activation='linear'))


    model.compile(optimizer=optimizer, loss=loss)

    return model


def create_dueldqn_network(input_shape, num_outputs):
        inputs = Input(shape=input_shape)

        net = Conv2D(16, 8, strides=(4, 4), activation='relu')(inputs)
        net = Conv2D(32, 4, strides=(2, 2), activation='relu')(net)

        net = Flatten()(net)
        advt = Dense(256, activation='relu')(net)
        advt = Dense(num_outputs)(advt)

        value = Dense(256, activation='relu')(net)
        value = Dense(1)(value)

        advt = Lambda(lambda advt: advt - K.mean(advt, axis=-1, keepdims=True))(advt)
        value = Lambda(lambda value: K.tile(value, [1, num_outputs]))(value)

        final = Add()([value, advt])

        model = Model(inputs=inputs, outputs=final)

        model.compile(
            optimizer=Adam(lr=0.001), 
            loss='mse'
        )
        return model
