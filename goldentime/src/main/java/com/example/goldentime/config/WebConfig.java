package com.example.goldentime.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.io.File;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        String projectPath = System.getProperty("user.dir");
        // 프로젝트 루트의 external-data/videos 폴더를 /videos/** URL로 매핑
        String uploadPath = "file:///" + projectPath + File.separator + "external-data" + File.separator + "videos" + File.separator;
        
        registry.addResourceHandler("/videos/**")
                .addResourceLocations(uploadPath);
    }
}
