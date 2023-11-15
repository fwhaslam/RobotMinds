import tensorflow as tf

first = None
val = (not first is None) and (first>1)

print('val=',val)

# four-d tensor with shape [2,2,2,3] filled with integers
# target1 = tf.expand_dims( tf.range(0, 24), axis=0 )
target1 = tf.expand_dims( tf.range(0, 4), axis=0 )
print("=",target1)
# target2 = tf.reshape( target1, (2,2,2,3) )
target2 = tf.reshape( target1, (1,2,2,1) )
print("target2=",target2)

target3 = tf.repeat( target2, repeats=2, axis=3 )
print("target3=",target3)
target4 = tf.repeat( target3, repeats=2, axis=2 )
print("target4=",target4)
print("target4.shape=",tf.shape(target4))



logits = [1.,2.,3.,4.]
tf.print('Logits=',logits)
logits = tf.nn.softmax( logits )
tf.print('Logits=',logits)
logits = tf.nn.softmax( logits )
tf.print('Logits=',logits)



model = tf.keras.Sequential()
model.add( tf.keras.layers.LSTM(4, input_shape=(1,10) ) )
model.add( tf.keras.layers.Dense(1) )
model.compile( loss='mse', optimizer='adam' )
model.summary()