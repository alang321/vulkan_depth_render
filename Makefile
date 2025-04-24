debug:
	mkdir -p build
	cd ./build && cmake -DCMAKE_BUILD_TYPE=Debug -DSAVE_IMAGES=OFF ../ && make -j7

release:
	mkdir -p build
	cd ./build && cmake -DCMAKE_BUILD_TYPE=Release -DSAVE_IMAGES=OFF ../ && make -j7

save_images:
	mkdir -p build
	cd ./build && cmake -DCMAKE_BUILD_TYPE=Debug -DSAVE_IMAGES=ON ../ && make -j7

clean:
	rm -rf build
	rm -rf bin
	find shaders -type f -name "*.spv" -delete
