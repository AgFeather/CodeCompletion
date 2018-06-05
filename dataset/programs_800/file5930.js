/* compile-metadata commander component
 * To use add require('../cmds/new-connection.js')(program) to your commander.js based node executable before program.parse
 */
'use strict';

var util                  = require('../../util').instance;
var OrgConnectionService  = require('../../org-connection');
var inherits              = require('inherits');
var BaseCommand           = require('../../command');

function Command() {
  Command.super_.call(this, Array.prototype.slice.call(arguments, 0));
}

inherits(Command, BaseCommand);

Command.prototype.execute = function() {
  var self = this;

  var project = self.getProject();
  var orgConnectionService = new OrgConnectionService(project);
  orgConnectionService.add(self.payload.username, self.payload.password, self.payload.orgType)
    .then(function() {
      self.respond('Org connection successfully created');
    })
    .catch(function(error) {
      self.respond('Could not add connection', false, error);
    })
    .done();  
};

exports.command = Command;
exports.addSubCommand = function(client) {
  client.program
    .command('new-connection [username] [password] [orgType]')
    .description('Adds a new deployment connection')
    .action(function(username, password, orgType){
      client.executeCommand(this._name, {
        username: username,
        password: password,
        orgType: orgType
      }); 
    });
};