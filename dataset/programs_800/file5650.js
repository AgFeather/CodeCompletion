casper.test.begin('test the mq-max-width-remove-class attribute', 4, function () {
	casper.start('http://127.0.0.1:9090/mq-max-width-remove-class', function () {
		casper.echo('Test on the screen with width size 480px', 'INFO');
		this.viewport(480, 480);
		this.wait(800, function () {
			casper.test.assertDoesntExist('.header-desktop', 'header-desktop should be removed');
			casper.test.assertDoesntExist('.content-desktop-1.content-desktop-2', 'Both the content-desktop-1 and content-desktop-2 should be removed');

			casper.echo('Test on the screen with the width size 481px.', 'INFO');
			this.viewport(481, 481);
			this.wait(800, function () {
				casper.test.assertExists('.header-desktop', 'header-desktop should be existed');
				casper.test.assertExists('.content-desktop-1.content-desktop-2', 'Both the content-desktop-1 and content-desktop-2 should be existed');
				casper.echo('Test Completed!!');					
			});
		});
	});

	casper.run(function () {
		casper.test.done();
	});
});