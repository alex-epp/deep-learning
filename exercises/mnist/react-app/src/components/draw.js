import React from 'react';
import ReactDOM from 'react-dom';

import {Predictor} from './model.js';


class BBox {
    constructor() {
        this.width = this.width.bind(this);
        this.height = this.height.bind(this);
        this.add = this.add.bind(this);
        this.clear = this.clear.bind(this);

        this.clear();
    }

    clear() {
        this.top = null;
        this.bottom = null;
        this.left = null;
        this.right = null;
    }

    add(x, y) {
        if (this.top == null || this.top > y) {
            this.top = y;
        }
        if (this.bottom == null || this.bottom < y) {
            this.bottom = y;
        }
        if (this.left == null || this.left > x) {
            this.left = x;
        }
        if (this.right == null || this.right < x) {
            this.right = x;
        }
    }

    width() {
        return this.right - this.left;
    }
    height() {
        return this.bottom - this.top;
    }
}

class LineGroup {
    constructor() {
        this.lines = []
        this.bbox = new BBox();

        this.empty = this.empty.bind(this);
        this.add = this.add.bind(this);
        this.clear = this.clear.bind(this);
        this.center = this.center.bind(this);
    }

    empty() {
        return this.lines.length === 0;
    }

    clear() {
        this.lines.length = 0;
        this.bbox.clear()
    }

    add(x1, y1, x2, y2) {
        this.lines.push([x1, y1, x2, y2]);
        this.bbox.add(x1, y1);
        this.bbox.add(x2, y2);
    }

    center() {
        if (this.empty())
            return null;

        var sx = 0, sy = 0;
        for(var i = 0; i < this.lines.length; i++) {
            sx += (this.lines[i][0] + this.lines[i][2]) / 2;
            sy += (this.lines[i][1] + this.lines[i][3]) / 2;
        }
        return [sx/this.lines.length, sy/this.lines.length];
    }
}

function lines_to_28x28(linegroup) {
    var canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    var ctx = canvas.getContext('2d');
    ctx.fillRect(0, 0, 28, 28);
    ctx.fillStyle="#000000";

    var c = linegroup.center();
    var r = Math.min(20/linegroup.bbox.width(), 20/linegroup.bbox.height());
    var x_offset = 14 - (c[0] - linegroup.bbox.left) * r;
    var y_offset = 14 - (c[1] - linegroup.bbox.top) * r;
    
    for(var i = 0; i < linegroup.lines.length; i++) {
        var x1 = x_offset + (linegroup.lines[i][0] - linegroup.bbox.left) * r;
        var y1 = y_offset + (linegroup.lines[i][1] - linegroup.bbox.top) * r;
        var x2 = x_offset + (linegroup.lines[i][2] - linegroup.bbox.left) * r;
        var y2 = y_offset + (linegroup.lines[i][3] - linegroup.bbox.top) * r;

        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
    }
    ctx.strokeStyle="white";
    ctx.lineWidth = 3;
    ctx.stroke();
    return ctx.getImageData(0, 0, 28, 28);
}


export class DrawingPadHolder extends React.Component {
    constructor(props) {
        super(props);

        this.last_x = null;
        this.last_y = null;
        this.is_mouse_down = false;
        this.ctx = null;
        this.canvas = null;
        this.model = new Predictor();
        this.lines = new LineGroup();
        this.state = {drawing: false, working: false, prediction: null}

        this.mouse_move = this.mouse_move.bind(this);
        this.mouse_leave = this.mouse_leave.bind(this);
        this.mouse_down = this.mouse_down.bind(this);
        this.mouse_up = this.mouse_up.bind(this);
        this.clear_clicked = this.clear_clicked.bind(this);
        this.predict_clicked = this.predict_clicked.bind(this);
        this.canvas_ref = this.canvas_ref.bind(this);
    }

    render() {
        var predict_class = "button card-footer-item";
        var clear_class = "button card-footer-item";
        if (this.state.drawing) {
            predict_class += " is-primary";
        }
        if (this.state.working) {
            predict_class += " is-loading";
        }
        return (
            <div className="column is-half is-offset-one-quarter">
                <div className="card" >
                    <div className="card-header">
                        <Prediction prediction={this.state.prediction} />
                    </div>
                    <div className="card-image">
                        <DrawingPad clear={() => {this.clear_clicked(null)}} canvas_resize={this.canvas_resize} canvas_ref={this.canvas_ref} mouse_move={this.mouse_move} mouse_leave={this.mouse_leave} mouse_down={this.mouse_down} mouse_up={this.mouse_up} points={this.state.points}/>
                    </div>
                    <div className="card-footer">
                        <a className={predict_class} disabled={!this.state.drawing} onClick={this.predict_clicked}>Predict</a>
                        <a className={clear_class} disabled={!this.state.drawing || this.state.working} onClick={this.clear_clicked}>Clear</a>
                    </div>
                </div>
            </div>
        );
    }

    canvas_ref(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
    }

    clear_clicked(e) {
        if (!this.state.drawing || this.state.working)
            return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.beginPath();
        this.setState({drawing: false, working: false, prediction: null});
        this.lines.clear();
    }
    predict_clicked(e) {
        if (!this.state.drawing || this.state.working)
            return;

        this.setState({working: true});
        
        var image_data = lines_to_28x28(this.lines);
        this.model.predict_from_image(image_data).then(function(pred) {
            this.setState({working: false, prediction: pred});
        }.bind(this), function(err) {
            console.log(err);
        });
    }

    draw_line(x, y) {
        if (this.state.working)
            return;
        
        this.lines.add(this.last_x, this.last_y, x, y);
        
        this.ctx.moveTo(this.last_x, this.last_y);
        this.ctx.lineTo(x, y);
        this.ctx.lineWidth = 3;
        this.ctx.stroke();
        this.setState({drawing: true});
    }

    mouse_move(x, y) {        
        if (this.last_x !== null && this.last_y !== null && this.is_mouse_down) {
            this.draw_line(x, y)
        }
        this.last_x = x
        this.last_y = y
    }

    mouse_leave(e) {
        this.last_x = null
        this.last_y = null
        this.is_mouse_down = false;
    }

    mouse_down(e) {
        this.is_mouse_down = true;
    }

    mouse_up(e) {
        this.last_x = null
        this.last_y = null
        this.is_mouse_down = false;
    }
}

function Prediction(props) {
    if (props.prediction == null) {
        return (
            <p className="card-header-title">Sketch a digit</p>
        )
    }
    else {
        return (
            <p className="card-header-title">Prediction: {props.prediction}</p>
        )
    }
}

class DrawingPad extends React.Component {
    constructor(props) {
        super(props);
        this.touch_move = this.touch_move.bind(this);
        this.mouse_move = this.mouse_move.bind(this);
        this.on_resize = this.on_resize.bind(this);
        this.state = {width: 0, height: 200}
    }

    componentDidMount() {
        window.addEventListener('resize', this.on_resize, false);
        ReactDOM.findDOMNode(this).addEventListener('touchstart', this.props.mouse_down, false);
        ReactDOM.findDOMNode(this).addEventListener('touchend', this.props.mouse_up, false);
        ReactDOM.findDOMNode(this).addEventListener('touchmove', this.touch_move, false);

        this.on_resize(null);
    }

    componentWillUnmount() {
        window.removeEventListener('resize', this.on_resize, false);
        ReactDOM.findDOMNode(this).removeEventListener('touchstart', this.props.mouse_down, false);
        ReactDOM.findDOMNode(this).removeEventListener('touchend', this.props.mouse_up, false);
        ReactDOM.findDOMNode(this).removeEventListener('touchmove', this.touch_move, false);
    }

    on_resize(e) {
        var rect = ReactDOM.findDOMNode(this).getBoundingClientRect();
        this.setState({width: rect.width, height: 200});
        this.props.clear();
    }

    render() {
        //<svg width="100%" height="100%" onMouseMove={this.mouse_move} onMouseLeave={this.props.mouse_leave} onMouseDown={this.props.mouse_down} onMouseUp={this.props.mouse_up}>
        //</svg>
        return (
            <canvas style={{width: "100%", height: "100%"}} width={this.state.width} height={this.state.height} ref={this.props.canvas_ref} onMouseMove={this.mouse_move} onMouseLeave={this.props.mouse_leave} onMouseDown={this.props.mouse_down} onMouseUp={this.props.mouse_up}>
                {this.props.points}
            </canvas>
        );
    }

    mouse_move(e) {
        var rect = ReactDOM.findDOMNode(this).getBoundingClientRect()
        var x_rel = e.clientX - rect.x
        var y_rel = e.clientY - rect.y
        this.props.mouse_move(x_rel, y_rel)
    }

    touch_move(e) {
        e.preventDefault();
        var rect = ReactDOM.findDOMNode(this).getBoundingClientRect()
        var x_rel = e.touches[0].clientX - rect.x
        var y_rel = e.touches[0].clientY - rect.y
        if (x_rel >= 0 && x_rel <= rect.width
                    && y_rel >= 0 && y_rel <= rect.height) {
            this.props.mouse_move(x_rel, y_rel);
        }
    }
}