const VADHandler = function (speakingEvent) {
    speakingEvent.detail ? (document.getElementById("VADLed").classList.add("led-red"), document.getElementById("VADLed").classList.remove("led-green")) : (document.getElementById("VADLed").classList.add("led-green"), document.getElementById("VADLed").classList.remove("led-red"))
}

const HotwordHandler = function (hotWordEvent) {
    hotword.pause()
    document.getElementById("LinTO").innerHTML = hotWordEvent.detail
    document.getElementById("LinTO").setAttribute('style', 'display:inline-block;')

    setTimeout(() => {
        hotword.resume()
        document.getElementById("LinTO").setAttribute('style', 'display:none;')
    }, 1500)
}
window.start = async function () {
    window.mic = new vitalWakeWord.Mic(JSON.parse(document.getElementById('mic').value))
    // window.downSampler = new vitalWakeWord.DownSampler()
    window.vad = new vitalWakeWord.Vad(JSON.parse(document.getElementById('VAD').value))
    // window.speechPreemphaser = new vitalWakeWord.SpeechPreemphaser()
    // window.feat = new vitalWakeWord.FeaturesExtractor()
    // window.hotword = new vitalWakeWord.Hotword()
    await mic.start()
    // await downSampler.start(mic)
    await vad.start(mic)
    // await speechPreemphaser.start(downSampler)
    // await feat.start(speechPreemphaser)
    // await hotword.start(feat, vad, 0.9)
    // await hotword.loadModel(hotword.availableModels["slinfox"])
    

    // new processing steps
    window.wakeWorkDownSampler = new vitalWakeWord.DownSampler({
        targetFrameSize: 1280,
        targetSampleRate: 16000,
        Int16Convert: true
        })
    await wakeWorkDownSampler.start(mic)

    window.wakeword = new vitalWakeWord.WakeWord()
    await wakeword.start(wakeWorkDownSampler)
    
    document.getElementById("VADLed").setAttribute('style', 'display:inline-block;')
    vad.addEventListener("speakingStatus", VADHandler)
    //hotword.addEventListener("hotword", HotwordHandler)
}

window.stop = async function () {
    // await downSampler.stop()
    await vad.stop()
    //await speechPreemphaser.stop()
    //await feat.stop()
    //await hotword.stop(feat, vad)
    
    await wakeWorkDownSampler.stop()
    await wakeword.stop()
    
    
    document.getElementById("VADLed").setAttribute('style', 'display:none;')
    vad.removeEventListener("speakingStatus", VADHandler)
}

window.rec = async function () {
    window.recMic = new vitalWakeWord.Recorder()
    window.recFeatures = new vitalWakeWord.Recorder()
    window.recDownsampler = new vitalWakeWord.Recorder()
    window.recSpeechPreemphaser = new vitalWakeWord.Recorder()
    window.recHw = new vitalWakeWord.Recorder()
    await recMic.start(mic)
    await recHw.start(hotword)
    await recFeatures.start(feat)
    await recDownsampler.start(downSampler)
    await recSpeechPreemphaser.start(speechPreemphaser)
    recMic.rec()
    recFeatures.rec()
    recDownsampler.rec()
    recSpeechPreemphaser.rec()
    recHw.rec()
}

window.stopRec = async function () {
    recMic.stopRec()
    recFeatures.stopRec()
    recDownsampler.stopRec()
    recSpeechPreemphaser.stopRec()
    recHw.stopRec()

    showLink(recMic)
    showLink(recFeatures)
    showLink(recDownsampler)
    showLink(recSpeechPreemphaser)
    showLink(recHw)
}

window.showLink = function (recInstance) {
    
    let url = recInstance.getFile()
    
    let link = window.document.createElement('a')
    
    link.href = url
    
    if (recInstance.hookedOn.type == "mic" || recInstance.hookedOn.type == "downSampler" || recInstance.hookedOn.type == "speechPreemphaser") {
        link.download = recInstance.hookedOn.type + ".wav"
    }
    
    if (recInstance.hookedOn.type == "featuresExtractor") {
        link.download = recInstance.hookedOn.type + '.json'
    }
    
    if (recInstance.hookedOn.type == "hotword") {
        link.download = recInstance.hookedOn.type + '.json'
    }
    
    link.textContent = recInstance.hookedOn.type
    
    let click = document.createEvent("Event")
    
    click.initEvent("click", true, true)
    
    link.dispatchEvent(click)
    
    // Attach the link to the DOM
    document.body.appendChild(link)
    
    let hr = window.document.createElement('hr')
    
    document.body.appendChild(hr)
}

// HTML Interface
document.getElementById('mic').value = JSON.stringify(vitalWakeWord.Mic.defaultOptions, false, 4)

document.getElementById('VAD').value = JSON.stringify(vitalWakeWord.Vad.defaultOptions, false, 4)

document.getElementById("start").onclick = async () => {
    start()
}

/*
document.getElementById("lintomodel").onclick = async () => {
    hotword.loadModel(hotword.availableModels["linto"])
}
*/

/*
document.getElementById("slinfoxmodel").onclick = async () => {
    hotword.loadModel(hotword.availableModels["slinfox"])
}
*/

document.getElementById("stop").onclick = async () => {
    stop()
}

document.getElementById("startrecord").onclick = async () => {
    rec()
}

document.getElementById("stoprecord").onclick = async () => {
    stopRec()
}