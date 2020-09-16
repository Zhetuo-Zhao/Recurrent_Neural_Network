# %%
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pdb
mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)


n_classes=10
batch_size=100

x=tf.placeholder('float',[None, 784])
y=tf.placeholder('float')

def conv2d(x,W,b,s=1):
    x=tf.nn.conv2d(x,W, strides=[1,s,s,1], padding='SAME')
    #pdb.set_trace()
    x=tf.nn.bias_add(x,b)
    x=tf.nn.relu(x)
    return x

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1],strides=[1,k,k,1], padding='SAME') 

def conv_network_model(data):
    weights={'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
             'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
             'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
             'out':tf.Variable(tf.random_normal([1024,n_classes]))}
   
    biases={'b_conv1':tf.Variable(tf.random_normal([32])),
             'b_conv2':tf.Variable(tf.random_normal([64])),
             'b_fc':tf.Variable(tf.random_normal([1024])),
             'out':tf.Variable(tf.random_normal([n_classes]))}

    x=tf.reshape(data,shape=[-1,28,28,1])
    conv1=maxpool2d(conv2d(x,weights['W_conv1'],biases['b_conv1']))
    conv2=maxpool2d(conv2d(conv1,weights['W_conv2'],biases['b_conv2']))
    
    fc=tf.reshape(conv2,[-1, 7*7*64])
    fc=tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    
    output=tf.matmul(fc, weights['out'])+biases['out']
    
    return output

def train_neural_network(x,y):
    prediction = conv_network_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer=tf.train.AdamOptimizer().minimize(cost)

    hm_epochs=10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c=sess.run([optimizer,cost],feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct= tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

        accuracy =tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
        
        
train_neural_network(x,y)