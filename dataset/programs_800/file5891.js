/**************************************************
* Learn Words // main.js
* coded by Anatolii Marezhanyi aka e1r0nd//[CRG] - March 2014
* http://linkedin.com/in/merezhany/ e1r0nd.crg@gmail.com
* Placed in public domain.
**************************************************/
// read settings
Settings.getSettings();

// set user saved local
if (local.currentLocal != $('[data-type=lang-select].selected').data('lang')) {
	$('[data-lang='+local.currentLocal+']').click();
};

// read vocabulary
var wordsIndex = localStorageAPI.readItem('learnWords-words').split(',');
Vocabulary.viewWord();
Learn.recountIndexLearn();
Learn.showWord();
Repeat.recountIndexRepeat();
Repeat.showWord();
Utils.closeMobMenu();