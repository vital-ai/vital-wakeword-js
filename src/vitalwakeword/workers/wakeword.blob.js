// <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>

self.importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js');

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

onmessage = async function (msg) {
    switch (msg.data.method) {
        case "configure":
            loadPyodideAndRunPython()
            break
        case "process":
            let result = await process(msg.data.audioFrame)
            
            // self.console.log('hey haley result: ', result);
    
            // TODO cleanup output
            
            let score = 0
            
            if(result != null && result != undefined) {
                
                let dict_value = result.get('hey_haley');
                
                if(dict_value != null && dict_value != undefined) {
                    
                    let array1 = dict_value[0];
                    
                    if(array1 != null && array1 != undefined) {
                        
                        let array2 = array1[0];
                        
                        if(array2 != null && array2 != undefined) {
                            

                            let array3 = array2[0];
                        
                        if(array3 != null && array3 != undefined) {
                            
                            if (typeof array3 === 'number') {
                            
                                // self.console.log('my hey haley score: ', array3);

                                score = array3;
                                
                            }
                            
                            if (Array.isArray(array3)) {
                            
                                let array4 = array3[0];
                                
                                if(array4 != null && array4 != undefined) {
                        
                                     if (typeof array4 === 'number') {
                            
                                        // self.console.log('my hey haley score: ', array4);

                                        score = array4;

                                         
                                    }
                                    
                                    if (Array.isArray(array4)) {
                                        
                                        
                                        let array5 = array4[0];

                                        
                                        if(array5 != null && array5 != undefined) {
                        
                                            if (typeof array5 === 'number') {
                            
                                                // self.console.log('my hey haley score: ', array5);
                                                
                                                score = array5;

                                                

                                            }
                                            else {
                                                // self.console.log('non-number my hey haley score: ', array5);
                                                
                                            }
                                            
                                                
                                        }
                                        
                                    }
                                    
                                    

                                }
                                

                            }
                            
                        }
                            
                            
                            
                        }
                        
                    }
                 
                    
                    
                    
                }
                

            }
            
            
            if(score > 0.1) {
                
                self.console.log('my hey haley score: ', score);

                postMessage({
                    type: 'wakeWordFrame',
                    score: score
                    });
            }
            
            break
    }
}

async function loadPyodideAndRunPython() {
    
    const pyodideModule = await import('https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.mjs');
    
    let pyodide = await pyodideModule.loadPyodide({
        indexURL : "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/"
    });
    
    await pyodide.loadPackage('numpy');
    
    // await pyodide.loadPackage('micropip');
        
    // await pyodide.runPythonAsync(`
    //     import micropip
    //     await micropip.install('cmudict')
    // `);
    
    async function loadPythonFile(pyodide, filePath) {
      const response = await fetch(filePath);
      const pythonCode = await response.text();
      pyodide.runPython(pythonCode);
    }

    const baseUrl = self.location.origin;

    await loadPythonFile(pyodide, `${baseUrl}/python/vad.py`);

    await loadPythonFile(pyodide, `${baseUrl}/python/init.py`);

    await loadPythonFile(pyodide, `${baseUrl}/python/utils.py`);

    await loadPythonFile(pyodide, `${baseUrl}/python/model.py`);
    
    let wakeWordModel = await pyodide.runPythonAsync(`

      model_path = '${baseUrl}/models/hey_haley.onnx'
      
      owwModel = Model(wakeword_models=[model_path])
    
      await owwModel.init()

      owwModel
    `);
        
     self.wakeWordModel = wakeWordModel;
        
  }

function squeezeArray(array) {
    
    let result = array;
    
    while (result.length === 1 && Array.isArray(result[0])) {
        result = result[0];
    }
    
    return result;
}
    
async function embeddingModelPredict(batch_count, x) {
    
    let session;
    
    
    try {
         
    let embedding_input = x.toJs();
    
    // self.console.log('embeddingModelPredict: ', embedding_input);
    
    const baseUrl = self.location.origin;

    session = await ort.InferenceSession.create(`${baseUrl}/models/embedding_model.onnx`);
    
    const tensor = new ort.Tensor('float32', embedding_input, [batch_count, 76, 32, 1]);

    const output = await session.run({ input_1: tensor })
        
    const outputTensor = output[Object.keys(output)[0]];
        
    // self.console.log('embedding output tensor: ', outputTensor);
        
    return outputTensor.cpuData;
        
    } catch (error) {
        
        console.error("An error occurred during inference:", error);
        
        throw error;    
        
    } finally {
        
            // free session & memory
            // non-public dispose call
         if (session) {
            session.handler.dispose(); 
          }
    }
}
    
async function melspecModelPredict(x) {
    
    let session;

    try {
    
    let mel_spec_input = x.toJs();
        
    const baseUrl = self.location.origin;
        
    session = await ort.InferenceSession.create(`${baseUrl}/models/melspectrogram.onnx`);

    const tensor = new ort.Tensor('float32', mel_spec_input , [1, mel_spec_input.length]);
    
    const output = await session.run({ input: tensor });
        
    const outputTensor = output[Object.keys(output)];
        
    // self.console.log('mel spec output tensor: ', outputTensor);
        
    return outputTensor.cpuData;
        
    } catch (error) {
        
        console.error("An error occurred during inference:", error);
        
        throw error;
        
    } finally {
        
        // free session & memory
            // non-public dispose call
         if (session) {
        
             session.handler.dispose();     
         }
    }
        
}
        

async function wakeWordPredict(x) {
    
    let session;

    try {
         
    const baseUrl = self.location.origin;

    session = await ort.InferenceSession.create(`${baseUrl}/models/hey_haley.onnx`);

    let input = x.toJs();
       
    const inputTensor = new ort.Tensor('float32', input, [1, 16, 96]);

    const outputMap = await session.run({ 'onnx::Flatten_0': inputTensor });

    const outputTensor = outputMap[Object.keys(outputMap)];
    
    return outputTensor.cpuData;
       
   } catch (error) {
        console.error("An error occurred during inference:", error);
        throw error;
    
    } finally {
        
        // free session & memory
            // non-public dispose call
        
         if (session) {
          
             session.handler.dispose();    
         }
    }         
}
  

async function process(audioFrame) {
        
    if(self.wakeWordModel == undefined) { return null; }
    
    let result = await self.wakeWordModel.predict_js(audioFrame);
    
    let result_js = result.toJs();
    
    let hey_haley = result_js.get('hey_haley');
    
    // self.console.log('processing audioFrame Result: ', result_js);
    
    // self.console.log('processing audioFrame Result HeyHaley: ', hey_haley[0][0]);
    
    return result_js;
}