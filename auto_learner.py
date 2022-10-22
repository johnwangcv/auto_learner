import flask
from markupsafe import escape
from flask import render_template
from flask import request
from flask import url_for
from flask import Flask
from flask import make_response
from flask import session

'''

sudo apt install -y python3
sudo apt install -y python3-pip

pip3 install torch --no-cache-dir
pip3 install transformers --no-cache-dir

pip3 install Flask

'''

from text_to_text_model import text_to_text_model

print('building the text to text model')

model = text_to_text_model()
model.build_model()

import sqlite3
con = sqlite3.connect("grammar.sqlite", check_same_thread=False)
cur = con.cursor()

import language_tool_python
tool = language_tool_python.LanguageToolPublicAPI('es') 

app = Flask(
	__name__
	)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

@app.route(
	'/spanish_grammar_checker',
	methods = [
		'POST',
		'GET',
		])

def spanish_grammar_checker():

	if request.method == 'POST':

		if 'input_text' in request.form:
			## model correction
			input_text = request.form['input_text']
			session.clear()
			session["input_text"] = input_text

			'''
			open source prediction
			'''

			try:
				corrected_text = tool.correct(input_text)
				matches = '<br>'.join([f'{m.context}:-> {m.message}' for m in tool.check(input_text)])
			except:
				corrected_text = ''
				matches = ''

			'''
			our model prediction
			'''
			corrected_text_our = model.inference(input_text)
			
			session["corrected_text"] = corrected_text

			#insert to db
			data = [
				(input_text, corrected_text,),
				]
			cur.executemany("INSERT INTO model_correction VALUES(?, ?)", data)
			con.commit()

			return render_template(
				"auto_learner.html",
				corrected_text = corrected_text,
				corrected_text_our = corrected_text_our,
				input_text = input_text,
				matches = matches,
				)

		if 'feedback_text' in request.form:
			## user feedback
			feedback_text = request.form['feedback_text']



			#insert to db
			data = [
				(session["input_text"], session["corrected_text"], feedback_text,),
				]
			cur.executemany("INSERT INTO user_feedback VALUES(?, ?, ?)", data)
			con.commit()

			#train the model

			'''
			training_set = [
				(r[0], r[2]) for r in cur.execute("SELECT * FROM user_feedback")
				]
			'''

			training_set = [
				(session["input_text"], feedback_text)
				]

			print(f'training model over {len(training_set)} training examples')

			model.train_model(
				training_set,
				epochs = 200
			)



		if 'good_quliaty_text' in request.form:
			## user feedback
			good_quliaty_text = request.form['good_quliaty_text']

			#insert to db
			data = [
				(good_quliaty_text, None, good_quliaty_text,),
				]
			cur.executemany("INSERT INTO user_feedback VALUES(?, ?, ?)", data)
			con.commit()

			#train the model

			'''
			training_set = [
				(r[0], r[2]) for r in cur.execute("SELECT * FROM user_feedback")
				]
			'''

			training_set = [
				(good_quliaty_text, good_quliaty_text),
				]

			print(f'training model over {len(training_set)} training examples')

			model.train_model(
				training_set,
				epochs = 200
			)


	return render_template(
		"auto_learner.html")


@app.route(
	'/user_feedback_db',
	methods = [
		'POST',
		'GET',
		])

def user_feedback_db():

	table_html = ''.join([
		f"""
			  <tr>
				<td>{r[0]}</th>
				<td>{r[1]}</th>
				<td>{r[2]}</th>
			  </tr>
		""" 
		for r in 
		cur.execute("SELECT * FROM user_feedback")
		])

	return f"""
		<html>

		<body>
		<h1>user feedback</h1>
		<br>
		<table border="1">
		<tr>
			<td>input text</th>
			<td>model correction</th>
			<td>user feedback</th>
		</tr>
		<tbody>
		{table_html}
		</tbody>
		</table>

		</body>
		</html>
		"""

@app.route(
	'/user_feedback_db_clear',
	methods = [
		'POST',
		'GET',
		])

def user_feedback_db_clear():
	cur.execute(u"""
		DROP TABLE user_feedback
		""")
	cur.execute(u"""
		CREATE TABLE user_feedback
		(
			input_text, 
			model_corrected_text,
			corrected_text
		)
		""")
	#con.commit() 

	return "table recreated."



'''

# start the service


rm -r -f auto_learner

git clone https://github.com/johnwangcv/auto_learner.git

cd auto_learner

nohup flask --app auto_learner --debug run --host=0.0.0.0 --port=3917 &

18.212.54.96:3917/spanish_grammar_checker

localhost:3917/spanish_grammar_checker



flask --app auto_learner --debug run --host=0.0.0.0 --port=3917

# user the service 

http://127.0.0.1:6912/spanish_grammar_checker

http://127.0.0.1:6912/user_feedback_db

http://127.0.0.1:6912/user_feedback_db_clear

'''
