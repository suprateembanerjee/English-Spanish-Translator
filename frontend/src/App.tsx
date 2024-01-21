import './App.css'

function postData(input:string) {

    fetch("http://localhost:5000/translate", {
      method: 'POST',
      headers: {'Content-Type' : 'application/json'},
      body: JSON.stringify({ 'text': input,
      'source_vectorization': 'models/english_vectorization.pkl', 
      'target_vectorization': 'models/spanish_vectorization.pkl', 
      'model': 'models/translator_transformer.keras'})
    }).then(res => res.json())
    .then(data => (document.getElementById('spanishText') as HTMLInputElement).value = data['translated'].substring(7, data['translated'].length - 5))
  }

function buttonHandler() {
  if ((document.getElementById('englishText') as HTMLInputElement).value != ''){
    postData((document.getElementById('englishText') as HTMLInputElement).value)
  }
  else {
    (document.getElementById('spanishText') as HTMLInputElement).value = ''
  }
}

function App() {

    return <div>
    <h1>English to Spanish <span className="me">Translator</span></h1>
    <textarea className='textarea1' id="englishText" rows={1} cols={33} placeholder='English Sentence'/>
    <button className="translateButton" role="button" onClick={buttonHandler}>Translate</button>
    <textarea className='textarea2' id="spanishText" rows={1} cols={33} placeholder='Spanish Sentence' readOnly={true}/>
    </div>
}

export default App