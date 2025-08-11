import React from 'react';

const HomePage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-8">
            NIS Protocol v3.2
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Neural Intelligence Synthesis - Agnostic Protocol Interface
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              System Status
            </h2>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Backend:</span>
                <span className="text-green-600">Connected</span>
              </div>
              <div className="flex justify-between">
                <span>Runner:</span>
                <span className="text-green-600">Connected</span>
              </div>
              <div className="flex justify-between">
                <span>Kafka:</span>
                <span className="text-green-600">Connected</span>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Available Providers
            </h2>
            <div className="space-y-2">
              <div className="text-gray-600">• OpenAI</div>
              <div className="text-gray-600">• Anthropic</div>
              <div className="text-gray-600">• Google</div>
              <div className="text-gray-600">• DeepSeek</div>
            </div>
          </div>
        </div>
        
        <div className="mt-12 bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">
            Protocol Information
          </h2>
          <p className="text-gray-600">
            NIS Protocol v3.2 provides a unified interface for multi-provider AI interactions.
            This deployment is running as an agnostic template for various project implementations.
          </p>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
