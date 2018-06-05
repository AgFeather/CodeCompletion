angular.module('app')
.run(function($rootScope, $location, $interval, Modal, DataManager, Pomodoro, PomodoroPersistence) {
  syncLocalPomodori()
  $interval(syncLocalPomodori,1000*60)

  $rootScope.$on('$stateChangeError', function (event, toState, toParams, fromState, fromParams, error) {
    console.log( '-- $stateChangeError :: error', error )
    if( toState.name.match(/^statistics/) ){
      Modal.show('loginPrompt')
    }
  })

  $rootScope.containsPath = function(checkPath){
    return $location.path().substr(1).indexOf(checkPath) >= 0
  }

  function syncLocalPomodori(){
    DataManager.popFrom('pomodoroSaveFailures').then(function(pomodoro){
      PomodoroPersistence.save(pomodoro).then(syncLocalPomodori)
    })
  }

})
