import Node from '../nodes/node.js'
import Worker from '../workers/wakeword.blob.js'
import NodeError from '../nodes/error.js'

const handler = function (nodeEvent) {
    this.workerRuntime.postMessage({
        method: "process",
        audioFrame: nodeEvent.detail
    })
}

export default class WakeWord extends Node {
    
    constructor() {
        super()
        this.worker = Worker
        this.handler = handler.bind(this)
        this.type = "wakeWord"
        this.event = "wakeWordFrame" //emitted
        this.hookableOnNodeTypes = ["downSampler"]
        
    }

    debounceTimer;
    
    playSound() {
        const audio = new Audio('/sounds/dingsound.mp3');
        audio.play();
    }
    
    handleEvent(event, threshold = 0.2, debounceTime = 1000){
        
        if (this.debounceTimer) return;
  
        if (event.score > threshold) {
            this.playSound();
            this.debounceTimer = setTimeout(() => {
                clearTimeout(this.debounceTimer);
                this.debounceTimer = null;
            }, debounceTime);
        }
    }
    
    async start(node){
        await super.start(node)
        this.workerRuntime.postMessage({
            method: "configure"
        })
        
        let _this = this;
        
        this.workerRuntime.onmessage = function(e) {
            
            if (e.data.type && e.data.type === 'wakeWordFrame') {
                
                let score = e.data.score;
                
                console.log('Wake Word Score:', score);
                
                _this.handleEvent({ score: score });
                
            }               
        }        
    }
}