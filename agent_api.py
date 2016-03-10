import cPickle
from flask import render_template
import json
from flask import Flask, request
import sys
from cnn_text_trainer.rw.datasets import clean_str
from string import Template
import threading
import logging
import properties
import json

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


#logging.basicConfig(level=properties.log_level,
 #                   format='%(asctime)s %(levelname)-8s %(message)s',
  #                  datefmt='%a, %d %b %Y %H:%M:%S',
   #                 filename=properties.log_file,
    #                filemode='w')

app = Flask(__name__)

class agentThread (threading.Thread):
	def __init__(self, threadID, name, model, title, content, aid, converted_text):
        	threading.Thread.__init__(self)
        	self.threadID = threadID
        	self.name = name
		self.model = model
		self.is_agent = '-1'
		self.title = title
		self.content = content
		self.converted_text = converted_text
		self.aid = aid
		self.label_to_prob = {}
    
	def run(self):
		try :
    			[y_pred,prob_pred] = self.model.classify([{'text':self.converted_text}])
    			labels = self.model.labels

    			agents = {}
    			agentIds = ""
    			for i in range(len(labels)):
				if labels[i] == '1' or labels[i] == '-1' :
        				self.label_to_prob[labels[i]]=prob_pred[0][i]
					if prob_pred[0][i] > 0.5 :
						self.is_agent = labels[i]	
			
			
    		except Exception, err:
        		app.logger.error( Exception, err)
			app.logger.error( "aid = " + self.aid)
			app.logger.error( "title = " + self.title)
			app.logger.error( "content = " + self.content  )
		

def repeat_to_length( string_to_expand, length):
	return (string_to_expand * ((length/len(string_to_expand))+1))[:length]

def convert_text( title, content ) :
	
	title = title + ". "
	title = title * 5

	news = title + content
	return repeat_to_length ( news, 8000)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
        app.logger.erro('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route('/test_agent')
def test_home () :
	return 'true'

@app.route('/agent_hit', methods=['POST'])
def agent_hit_home () :
	
	try :
		title = request.args.get('title').lower()
        	content = request.args.get('content').lower()
        	aid=request.args.get('aid').lower()
		converted_text = convert_text ( title, content)
		converted_text = clean_str(converted_text)
	        app.logger.info('text received')
                app.logger.info(title)
                app.logger.info(content)
                app.logger.info(aid)
	
		lc_thread = agentThread(1, '1', lc_model, title, content, aid, converted_text)
		no_thread = agentThread(2, '2', no_model, title, content, aid, converted_text)
		ac_thread = agentThread(3, '3770', ac_model, title, content, aid, converted_text)
		pa_thread = agentThread(4, '29', pa_model, title, content, aid, converted_text)
		
                app.logger.info('threads creates')
                lc_thread.start()
		no_thread.start()
		ac_thread.start()
		pa_thread.start()
                app.logger.info('threads started')
		
		threads = []
		threads.append(lc_thread)
		threads.append(no_thread)
		threads.append(ac_thread)
		threads.append(pa_thread)

		for t in threads :
			t.join()
		
		agents = {}
		
		agents['aid'] = aid

		for t in threads :
			if t.is_agent == '1':
				agents[t.name] = 'true'
			else :
				agents[t.name] = 'false'
			agents[t.name + '_probs'] = t.label_to_prob
             
                temp_str = str(agents)   
                app.logger.info ( aid + '\t' + temp_str)
		return json.dumps(agents)

    	except Exception as e:
                logging.exception("read error")
        	app.logger.error( "Error processing request. Improper format of request.args['text'] might be causing an issue. Returning empty array")
       		return json.dumps({})


port=8983
debug = True
load_word_vecs = True
    
lc_model_path = properties.lc_model_path
no_model_path = properties.no_model_path
ac_model_path = properties.ac_model_path
pa_model_path = properties.pa_model_path

#In memory dictionary which will load all the models lazily
#In memory dictionary which will load all the models lazily
lc_model = cPickle.load(open(lc_model_path,"rb"))
no_model = cPickle.load(open(no_model_path,"rb"))
ac_model = cPickle.load(open(ac_model_path,"rb"))
pa_model = cPickle.load(open(pa_model_path,"rb"))

logHandler = logging.FileHandler(properties.log_file)
logHandler.setLevel(properties.log_level)
app.logger.addHandler(logHandler)
app.logger.setLevel(properties.log_level)

if __name__ == "__main__":
    #In memory dictionary which will load all the word vectors lazily
    wordvecs={}

    app.run(debug=debug,host='0.0.0.0',port=port, threaded=True)
