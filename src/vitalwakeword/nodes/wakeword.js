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

                const event = new CustomEvent('wakeWordEvent', 
                                { detail: score });

                window.dispatchEvent(event);
                
            }               
        }        
    }
}