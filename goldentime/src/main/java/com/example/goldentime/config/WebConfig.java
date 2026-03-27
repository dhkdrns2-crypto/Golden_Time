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
        // 프로젝트 루트의 external-data 폴더들을 URL로 매핑
        String videosPath = "file:///" + projectPath + File.separator + "external-data" + File.separator + "videos" + File.separator;
        String imagesPath = "file:///" + projectPath + File.separator + "external-data" + File.separator + "images" + File.separator;
        
        registry.addResourceHandler("/videos/**")
                .addResourceLocations(videosPath);

        // 공지사항 업로드 이미지는 여기로 서빙
        registry.addResourceHandler("/images/**")
                .addResourceLocations(imagesPath);
    }
}
