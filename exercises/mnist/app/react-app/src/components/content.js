import React from 'react';
import 'bulma/css/bulma.css';
import {DrawingPadHolder} from './draw.js'


export function Header(props) {
    return (
      <header className="hero is-medium is-primary is-bold">
        <div className="hero-body">
          <div className="container">
            <h1 className="title">
              MNIST Digit Classifier
            </h1>
            <SourceButton />
          </div>
        </div>
      </header>
    );
  }
  
  export function Body(props) {
    return (
      <section className="section">
        <div className="container">
          <div className="content has-text-centered">
            <DrawingPadHolder />
          </div>
          <div className="content">
            <Description />
          </div>
        </div>
      </section>
    );
  }
  
  export function Footer(props) {
    return (
      <footer className="footer">
        <div className="container">
          <div className="content has-text-centered">
            <p>
              By <a href="https://github.com/alex-epp">Alexander Epp</a>
            </p>
          </div>
        </div>
      </footer>
    );
  }
  
  function SourceButton(props) {
    return (
      <a className="button is-primary" href="https://github.com/alex-epp/deep-learning/tree/master/exercises/mnist">
          See the source at GitHub/alex-epp
        </a>
    );
  }
  
  function Description(props) {
    return (
      <div>
        <h1 className="title is-3">
          What is this?
        </h1>
        <p>
          This is a demonstration of a neural network library I created from scratch (i.e. no Tensorflow or similar libaries). I have trained the model to recognize single digits drawn by a user.
        </p>
        <br />
        <h1 className="title is-3">
          How does it work?
        </h1>
        <p>
          The model consists of five individually-trained MLPs, each trained on the MNIST dataset for 10 epochs with minibatch size 200. Each MLP consists of an input layer of 784 units (one for each pixel), a hidden layer of 784 units (relu activation), and a 10-unit output layer (softmax activation).
        </p>
        <br />
        <p>
          When an image is written by the user, the software first preprocesses it similarly to how the MNIST data were preprocessed (converts it to white on black, shrinks it to 20px-20px, centers its center-of-mass on a black 28px-28px image), then obtains a prediction from each of the five models, which are averaged into the final prediction.
        </p>
        <br />
      </div>
    );
  }